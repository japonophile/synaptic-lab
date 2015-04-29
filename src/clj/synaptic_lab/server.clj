(ns
  ^{:doc "Experimenting with, and visualizing Synaptic neural networks (server)"
    :author "Antoine Choppin"}
  synaptic-lab.server
  (:require [synaptic.core :refer :all]
            [synaptic.net :refer [weight-histograms]]
            [synaptic.util :refer :all]
            [clatrix.core :as m]
            [clojure.core.async :as async :refer [<! <!! chan go thread]]
            [compojure.core :refer [defroutes GET POST routes]]
            [compojure.handler :as h]
            [compojure.route :as r]
            [org.httpkit.server :as kit]
            [taoensso.sente :as s]
            [taoensso.sente.server-adapters.http-kit
             :refer (sente-web-server-adapter)])
  (:import  [java.io File]
            [synaptic.net.Net])
  (:gen-class))

(def chunk-size 100)

(defn net-to-map-no-weights
  [nn]
  {:header (:header nn)
   :arch (into {} (:arch nn))
   :weights []
   :training (assoc (into {} (:training nn))
                    :state { :state (-> nn :training :state :state) } ;for now
                    :stats (into {} (-> nn :training :stats)))
   :save-state (:save-state nn)})

(defn net-to-map
  [nn]
  (assoc (net-to-map-no-weights nn) :weights (vec (map m/dense (:weights nn)))))

(defn hist-to-map
  [hist]
  {:labels (:labels hist) :data (:data hist)})

(defn load-net
  [net-name]
  (atom (assoc (load-data "neuralnet" net-name) :save-state :saved)))

(defn load-trset!
  [app-state trset-key]
  (println (str "loading... " (name trset-key)))
  (swap! app-state assoc-in [:trsets trset-key] (load-training-set (name trset-key)))
  (println "done."))

(defn save-net!
  [net]
  (let [nn (dissoc @net :save-state)]
    (save-neural-net nn)
    (swap! net assoc :save-state :saved)))

(defn scale-weights
  "Scaling weights between 0 and 255 for display as grayscale."
  [weights]
  (let [minw  (apply min (map #(apply min (rest %)) weights))
        maxw  (apply max (map #(apply max (rest %)) weights))]
    (mapv (fn [ws]
            (mapv #(int (Math/round (* 255 (/ (- % minw) (- maxw minw)))))
                  (drop 1 ws)))
          weights)))

(defn scale-net-weights
  "Scale network weights."
  [nn]
  (assoc nn :weights (mapv scale-weights (:weights nn))))

(defn trset-to-map
  [trset]
  {:header (:header trset)})

(def app-state (atom {:nets {} :trsets {}}))

(doseq [nnname (file-list "neuralnet")]
  (swap! app-state assoc-in [:nets (keyword nnname)] (load-net nnname)))

(doseq [tsname (file-list "trainingset-header")]
  (println tsname)
  (swap! app-state assoc-in [:trsets (keyword tsname)]
                            {:header (load-training-set-header tsname)}))

(let [{:keys [ch-recv send-fn ajax-post-fn
              ajax-get-or-ws-handshake-fn] :as sente-info}
      (s/make-channel-socket! sente-web-server-adapter {})]
  (def ring-ajax-post   ajax-post-fn)
  (def ring-ajax-get-ws ajax-get-or-ws-handshake-fn)
  (def ch-chsk          ch-recv)
  (def chsk-send!       send-fn))

(defn public-root
  "Return the absolute (root-relative) version of the public path."
  []
  (str (System/getProperty "user.dir") "/resources/public"))

(defn unique-id
  "Return a really unique ID (for an unsecured session ID).
  No, a random number is not unique enough. Use a UUID for real!"
  []
  (rand-int 10000))

(defn session-uid
  "Convenient to extract the UID that Sente needs from the request."
  [req]
  (get-in req [:session :uid]))

(defn index
  "Handle index page request."
  [req]
  {:status 200
   :session (if (session-uid req)
              (:session req)
              (assoc (:session req) :uid (unique-id)))
   :body (slurp (str (public-root) "/index.html"))})

(defn send-net!
  [net-key uid state]
  (let [net-to-send (-> state :nets net-key
                        deref net-to-map scale-net-weights)]
    (chsk-send! uid [:synaptic/update-state [[:nets net-key] net-to-send]])))

(defn send-net-no-weights!
  [net-key uid state]
  (let [net-to-send (-> state :nets net-key deref net-to-map-no-weights)]
    (chsk-send! uid [:synaptic/update-state [[:nets net-key] net-to-send]])))

(defn send-net-training-stats!
  [net-key uid state]
  (let [net-to-send (-> state :nets net-key deref net-to-map-no-weights)]
    (chsk-send! uid [:synaptic/update-state
                     [[:nets net-key :training :stats]
                      (-> net-to-send :training :stats)]])))

(defn send-net-training-state!
  [net-key uid state]
  (let [net-to-send (-> state :nets net-key deref net-to-map-no-weights)]
    (chsk-send! uid [:synaptic/update-state
                     [[:nets net-key :training :state]
                      (-> net-to-send :training :state)]])))

(defn send-net-save-state!
  [net-key uid state]
  (let [net-save-state (-> state :nets net-key deref :save-state)]
    (println (str "send save-state " net-save-state))
    (chsk-send! uid [:synaptic/update-state
                     [[:nets net-key :save-state] net-save-state]])))

(defn send-net-weights-updated!
  [net-key uid]
  (chsk-send! uid [:synaptic/update-state [[:nets net-key :weights-updated] true]]))

(defn send-net-weights!
  [net-key uid state]
  (let [net-to-send (-> state :nets net-key deref net-to-map scale-net-weights)]
    (chsk-send! uid [:synaptic/update-state
                     [[:nets net-key :weights] (:weights net-to-send)]])))

(defn send-net-weight-histograms!
  [net-key uid state]
  (let [weights     (-> state :nets net-key deref :weights)
        weight-hist (mapv hist-to-map (weight-histograms weights))]
    (chsk-send! uid [:synaptic/update-state
                     [[:nets net-key :weight-hist] weight-hist]])))

(defn send-nets-no-weights!
  [uid state]
  (doseq [net-key (-> state :nets keys)]
    (send-net-no-weights! net-key uid state)))

(defn send-nets-weights!
  [uid state]
  (doseq [net-key (-> state :nets keys)]
    (send-net-weights! net-key uid state)
    (send-net-weight-histograms! net-key uid state)))

(defn send-trset-header!
  [trset-key uid state]
  (let [trset-to-send (-> state :trsets trset-key trset-to-map)]
    (chsk-send! uid [:synaptic/update-state [[:trsets trset-key] trset-to-send]])))

(defn send-trsets-header!
  [uid state]
  (doseq [trset-key (-> state :trsets keys)]
    (send-trset-header! trset-key uid state)))

(defn send-state!
  [uid state]
  (send-nets-no-weights! uid state)
  (send-trsets-header! uid state)
  ;(send-nets-weights! uid state)   ; net weights will only be sent on demand
  )

(defn weights-differ
  "Returns true if the 2 sets of weights differ by at least one value.
  This function was needed, because the equiv method of clatrix.core will
  not work properly on matrices (sometimes return true while some values
  are in fact different)."
  [w1 w2]
  (or (not= w1 w2)
      (some true? (map #(some true? (map not= (flatten %1) (flatten %2))) w1 w2))))

(defn watch-net-state
  [net-key uid]
  (fn [k net oldnn newnn]
    (if-not (= :modified (-> newnn :save-state))
      (swap! net assoc :save-state :modified))
    (when (not= (-> oldnn :training :state) (-> newnn :training :state))
      (println (str "training state updated " (-> newnn :training :state :state)))
      (send-net-training-state! net-key uid @app-state))
    (when (not= (-> oldnn :training :stats) (-> newnn :training :stats))
      (println "training stats updated")
      (send-net-training-stats! net-key uid @app-state))
    (when (weights-differ (:weights oldnn) (:weights newnn))
      (println "weights updated")
      (send-net-weights-updated! net-key uid))
    (when (not= (-> oldnn :save-state) (-> newnn :save-state))
      (println (str "save-state updated " (-> newnn :save-state)))
      (send-net-save-state! net-key uid @app-state))))

(defmulti handle-event
  "Handle events based on the event ID."
  (fn [[ev-id ev-arg] ring-req] ev-id))

(defmethod handle-event
  :synaptic/refresh-state
  [[_ path] req]
  (when-let [uid (session-uid req)]
    (cond
      (empty? path)
        (send-state! uid @app-state)
      (= (assoc path 1 nil) [:nets nil :weights])
        (let [net-key (nth path 1)]
          (send-net-weights! net-key uid @app-state)
          (send-net-weight-histograms! net-key uid @app-state))
      :else
        (println (str "unsupported path " path " for refresh-state")))))

(defmethod handle-event
  :synaptic/create-net
  [[_ [net-key net-def]] req]
  (if-let [uid (session-uid req)]
    (let [header (-> net-def :header)
          layers (-> net-def :arch :layers)
          algo   (-> net-def :training :algo)
          params (-> net-def :training :params)
          net    (neural-net layers (training algo params))]
      (swap! net assoc :header header)
      (swap! app-state assoc-in [:nets net-key] net)
      (save-net! net)
      (send-net! net-key uid @app-state))))

(defmethod handle-event
  :synaptic/save-net
  [[_ net-key] req]
  (if-let [uid (session-uid req)]
    (when-let [net (-> @app-state :nets net-key)]
      (save-net! net)
      (send-net-save-state! net-key uid @app-state))))

(defmethod handle-event
  :synaptic/start-training
  [[_ [net-key trset-key nepochs]] req]
  (when-let [uid (session-uid req)]
    (let [net   (-> @app-state :nets net-key)
          trset (-> @app-state :trsets trset-key)
          tsname (name trset-key)]
      (println (str "training with " tsname))
      (if (nil? (:batches trset))
        (load-trset! app-state trset-key))
      (add-watch net :train-ui (watch-net-state net-key uid))
      @(train net (-> @app-state :trsets trset-key) nepochs)
      (remove-watch net :train-ui))))

(defmethod handle-event
  :synaptic/stop-training
  [[_ [net-key]] req]
  (when-let [uid (session-uid req)]
    (let [net (-> @app-state :nets net-key)]
      (stop-training net))))

(defmethod handle-event
  :synaptic/fetch-trset-chunk
  [[_ [set-key t i chunk-start]] req]
  (when-let [uid (session-uid req)]
    (if (nil? (-> @app-state :trsets set-key :batches))
      (load-trset! app-state set-key))
    (let [data-set      (if (= :batch t)
                          (nth (-> @app-state :trsets set-key :batches) i)
                          (-> @app-state :trsets set-key :valid))
          data-set-smp  (-> data-set :x m/dense)
          uniquelabels  (-> @app-state :trsets set-key :header :labels)
          data-set-lb   (if (:y data-set)
                          (frombinary uniquelabels
                            (mapv (partial mapv int) (-> data-set :y m/dense))))
          chunk-samples (vec (take chunk-size (drop chunk-start data-set-smp)))
          chunk-labels  (if data-set-lb
                          (vec (take chunk-size (drop chunk-start data-set-lb))))]
      (chsk-send! uid [:synaptic/update-trset-chunk
                       [set-key t i chunk-start [chunk-samples chunk-labels]]]))))

(defmethod handle-event
  :synaptic/test-net
  [[_ [net-key set-key]] req]
  (when-let [uid (session-uid req)]
    (let [nn @(-> @app-state :nets net-key)
          ts (-> @app-state :trsets set-key)]
      (if (nil? (:batches ts))
        (load-trset! app-state set-key))
      (let [ts  (-> @app-state :trsets set-key)
            out {:batches (mapv #(estimate nn %) (:batches ts))
                 :valid (if (:valid ts) (estimate nn (:valid ts)))}]
        (chsk-send! uid [:synaptic/update-state
                         [[:nets net-key :testing :result] out]])))))

(defmethod handle-event
  :chsk/ping
  [_ req]
  (when-let [uid (session-uid req)]
    (chsk-send! uid [:synaptic/pong])))

(defmethod handle-event
  :default
  [event req]
  nil)

(defn event-loop
  "Handle inbound events."
  []
  (go (loop [{:keys [client-uuid ring-req event] :as data} (<! ch-chsk)]
        (println "-" event)
        (thread (handle-event event ring-req))
        (recur (<! ch-chsk)))))

(defroutes server
  (-> (routes
       (GET  "/"   req (#'index req))
       (GET  "/ws" req (#'ring-ajax-get-ws req))
       (POST "/ws" req (#'ring-ajax-post   req))
       (r/resources "/")
       (r/not-found "<p>Page introuvable.  Quelle tristesse!</p>"))
      h/site))

(defn -main
  "Start the http-kit server. Takes no arguments.
  Environment variable PORT can override default port of 3000."
  [& args]
  (event-loop)
  (let [port (or (System/getenv "PORT") 3000)]
    (println "Public root is" (public-root))
    (println "Starting Sente server on port" port "...")
    (kit/run-server #'server {:port port})))


