(ns
  ^{:doc "Experimenting with, and visualizing Synaptic neural networks (client)"
    :author "Antoine Choppin"}
  synaptic_lab.client
  (:require-macros [cljs.core.async.macros :refer [go]])
  (:require [om.core :as om :include-macros true]
            [om.dom :as dom :include-macros true]
            [cljs.core.async :as async :refer [<! >! put! chan]]
            [taoensso.sente :as s :refer (cb-success?)]))

(enable-console-print!)

(defn default-training-params
  [algo]
  (assoc
    ({ :perceptron { :error-fn :misclassification
                     :stats { :errorkinds [:misclassification] } }
       :backprop   { :error-fn :cross-entropy
                     :learning-rate { :epsilon 0.01 }
                     :stats { :errorkinds [:misclassification :cross-entropy] } }
       :rprop      { :error-fn :cross-entropy
                     :rprop { :mins 0.000001 :maxs 50.0 }
                     :stats { :errorkinds [:misclassification :cross-entropy] } }
       :rmsprop    { :error-fn :cross-entropy
                     :rmsprop { :alpha 0.9 }
                     :learning-rate { :epsilon 0.01 }
                     :stats { :errorkinds [:misclassification :cross-entropy] } }
       :lbfgs      { :error-fn :cross-entropy
                     :stats { :errorkinds [:misclassification :cross-entropy] } }
     } algo)
    :trset nil))

(defn default-net
  [algo]
  { :header
    { :name "No name" }
    :arch
    { :layers [{:type :input :fieldsize [1 0 0]}
               {:type :fully-connected :n 0
                :act-fn (if (= algo :perceptron) :binary-threshold :sigmoid)}]
    }
    :training
    { :algo algo
      :params (default-training-params algo)
    }
    :save-state :unsaved
  })

(def chunk-size 100)

(defn trset-fsize
  [trset]
  (let [fsize (-> trset :header :fieldsize)]
    (if (= 3 (count fsize)) fsize (into [1] fsize))))

(defn init-new-net-trset!
  [app]
  (if (and (nil? (-> @app :nets :new :training :params :trset))
           (< 0 (count (-> @app :trsets))))
    (let [n         (-> @app :nets :new :arch :layers count)
          trset-key (-> @app :trsets keys first)
          trset     (-> @app :trsets first second)
          insize    (trset-fsize trset)
          outsize   (-> trset :header :labels count)]
      (om/transact! app [:nets :new :training :params :trset]   (fn [_] trset-key))
      (om/transact! app [:nets :new :arch :layers 0 :fieldsize] (fn [_] insize))
      (om/transact! app [:nets :new :arch :layers (dec n) :n]   (fn [_] outsize)))))

(def training-algorithms
  #{:perceptron :backprop :rprop :rmsprop :lbfgs})

(def activation-functions
  #{:linear :binary-threshold :binary-stochastic :hyperbolic-tangent :sigmoid :softmax})

(def layer-types
  #{:fully-connected :convolution})

(let [{:keys [chsk ch-recv send-fn state]}
      (s/make-channel-socket! "/ws" {:type :auto})]
  (def chsk       chsk)
  (def ch-chsk    ch-recv)  ; ChannelSocket's receive channel
  (def chsk-send! send-fn)  ; ChannelSocket's send API fn
  (def chsk-state state))   ; Watchable, read-only atom

(def app-state
  "The application state, yet to be initialized"
  (atom {:nets {:new (default-net :backprop)} :sel {:net nil} :trsets {}}))

(defn divisors
  "Returns 2 divisors of n that give the biggest product"
  [n]
  (loop [d (int (Math/sqrt n))]
    (cond
      (= 0 d)         [n 1]
      (= 0 (mod n d)) [(/ n d) d]
      :else           (recur (dec d)))))

;  (let [start (.getTime (js/Date.))]                                         ;;;
;  (.log js/console (str "task took " (- (.getTime (js/Date.)) start) "ms"))) ;;;

(defn display
  "Returns the Javascript style for an element to display or not"
  [show]
  (if show
    #js {}
    #js {:display "none"}))

(defn loading-icon
  [_ owner]
  (om/component
    (dom/img #js {:style #js {:display "block" :margin "auto"}
                  :src "/assets/wait20.gif"} nil)))

(defn select-net!
  [sel net-key]
  (om/transact! sel [:net] (fn [_] net-key)))

(defn init-net-param
  [net ks default-val]
  (assoc-in net ks (get-in net ks default-val)))

(defn compatible?
  [net trset]
  (let [insize  (-> net :arch :layers first :fieldsize)
        outsize (-> net :arch :layers last :n)
        labels  (-> trset :header :labels)]
    (and (= insize (trset-fsize trset))
         (or (nil? labels) (= outsize (count labels))))))

(defn selected-net
  [app]
  (if-let [sn (-> app :sel :net)]
    (if-let [net (-> app :nets sn)]
      (-> net (init-net-param [:header :name] (name sn))))))

(defn select-trset!
  [sel trset-key]
  (om/transact! sel [:training :params :trset] (fn [_] trset-key)))

(defn refresh-state!
  "Refresh the application state asynchronously"
  ([app path]
   (chsk-send! [:synaptic/refresh-state path]))
  ([app] (refresh-state! app [])))

(defn refresh-net-weights!
  [app net-key]
  (refresh-state! app [:nets net-key :weights])
  (om/transact! app [:nets net-key] (fn [net] (dissoc net :weights-updated))))

(defn net-list-view
  "Om component that displays the list of networks, and allow selecting
  a network to be displayed"
  [[sel nets] owner]
  (om/component
    (dom/div #js {:className "net-list panel"}
      (apply dom/ul nil
        (map #(dom/li #js {:onClick (fn [_] (select-net! sel %))
                           :className (if (= (:net sel) %) "selected" "")}
                      (if (= :new %)
                        "New..."
                        (or (-> nets % :header :name) (name %))))
             (conj (vec (keys (dissoc nets :new))) :new))))))

(def net-perspectives {:arch "Architecture" :weights "Weights"
                       :training "Training" :testing "Testing"})

(defn net-navigation-bar
  "Om component that displays the navigation bar, that is, a way to select a
  particular perspective for the currently selected neural net"
  [net owner {:keys [ch]}]
  (reify
    om/IInitState
    (init-state [this]
      {:perspective :arch})
    om/IRenderState
    (render-state [this {:keys [perspective]}]
      (dom/div #js {:className "navig-bar"}
        (apply dom/ul nil
          (map #(dom/li #js {:onClick (fn [_] (put! ch [:perspective %]))
                             :className (if (= perspective %) "selected" "")}
                        (net-perspectives %))
               (keys net-perspectives)))))))

(defmulti net-perspective-view
  "Om component (multi-method parametrized by the current view) that displays
  the current view of the selected neural net.  This component is specialized
  to display each kind of view: architecture, weights or training"
  (fn [[perspective _] _] perspective))

(defmulti layer-nunits-as-str
  (fn [layer] (:type layer)))

(defmethod layer-nunits-as-str
  :input
  [layer]
  (let [[k w h] (:fieldsize layer)]
    (str k " x " w " x " h)))

(defmethod layer-nunits-as-str
  :convolution
  [layer]
  (let [fmap  (:feature-map layer)
        k     (:k fmap)
        [w h] (:size fmap)
        pool  (:pool layer)]
    (str k " x " w " x " h
         (if pool
           (let [[pw ph] (:size pool)]
             (str ", pool " pw " x " ph ""))))))

(defmethod layer-nunits-as-str
  :fully-connected
  [layer]
  (str (:n layer) " units"))

(defn layer-actfn-as-str
  [layer]
  (if-let [actfn (:act-fn layer)]
    (str (name actfn) " activation function")))

(defn layer-description
  [layer]
  (str (layer-nunits-as-str layer)
       (if (not= :input (:type layer))
         (str " (" (layer-actfn-as-str layer) ")")
         "")
       " fieldsize: " (pr-str (:fieldsize layer))))

(defn net-architecture-view
  [arch owner]
  (let [layers (:layers arch)
        n      (count layers)
        nunits (mapv :n layers)
        act-fn (mapv :act-fn layers)]
    (om/component
      (dom/div #js {:className "net-arch-view"}
        (dom/h4 nil "Network Architecture")
        (dom/span nil (str "Network with " n " layers (1 input layer"
                           (cond
                             (= n 3) (str ", 1 hidden layer")
                             (> n 3) (str ", " (- n 2) " hidden layers")
                             :else   "")
                           " and 1 output layer)"))
        (apply dom/ul nil
          (map #(dom/li nil 
            (let [l (nth layers %)]
              (cond
                (= % 0)       (str "Input layer: " (layer-description l))
                (= % (dec n)) (str "Output layer: " (layer-description l))
                :else         (str "Hidden layer " % ": " (layer-description l)))))
            (range n)))))))

(defmethod net-perspective-view
  :arch
  [[_ app] owner]
  (om/component
    (dom/div #js {:className "net-view"}
      (om/build net-architecture-view (:arch (selected-net app))))))

(defn draw-pixels!
  "Draw pixels on the graphic context of an HTML canvas with a given width and height"
  [ctx w h colors]
  (let [id (.createImageData ctx 1 1)
        d  (doto (.-data id) (aset 3 255))]
    (doall (for [x (range w)
                 y (range h)]
             (let [c (nth colors (+ x (* w y)))]
               (aset d 0 c)
               (aset d 1 c)
               (aset d 2 c)
               (.putImageData ctx id x y))))))

(defn draw-neuron
  [weights w h owner]
  (let [ctx (.getContext (om/get-node owner) "2d")]
    (draw-pixels! ctx w h weights)))

(defn neuron-view
  "Om component that displays weights of a particular neuron"
  [[weights [w h]] owner]
  (let [[wp hp] (mapv #(str % "px") [w h])]
    (reify
      om/IRender
      (render [this]
        (dom/canvas #js {:className "neuron"
                         :style #js {:width wp :height hp}
                         :width wp :height hp} nil))
      om/IDidUpdate
      (did-update [this prev-props prev-state]
        (draw-neuron weights w h owner))
      om/IDidMount
      (did-mount [this]
        (draw-neuron weights w h owner)))))

(defn layer-weights-view
  "Om component that displays all the weights of a given network layer"
  [[layer-weights layer] owner]
  (om/component
    (let [kwh   (-> layer-weights first count)
          [w h] (if-let [featuremap (:feature-map layer)]
                  (:size featuremap)
                  (divisors kwh))
          wh    (* w h)
          k     (quot kwh wh)]
      (prn kwh [w h] wh k)
      (apply dom/div #js {:className "layer-view panel"}
        (doall (for [i (range k)]
          (apply dom/div nil
            (om/build-all neuron-view
                          (map #(vector (take wh (drop (* i wh) %1)) %2)
                               layer-weights
                               (repeat [w h]))))))))))

(defn make-data-series
  "Returns a data series by attaching the x axis to y-data"
  [y-axis & [x-axis]]
  (mapv (fn [a b] {:y a :x b}) y-axis (or x-axis (rest (range)))))

(defn nv-clear-graph
  [elem]
  (.. js/d3 (select elem) (select "svg") (remove))
  (.. js/d3 (select elem) (append "svg")))

(defn nv-bar-graph
  "Draw a bar graph (histogram) using NVD3"
  [elem hist-data]
  (let [chart (.. js/nv -models multiBarChart
                  (transitionDuration 350)
                  (showLegend false)
                  (showControls false)
                  (showYAxis true)
                  (showXAxis true)
                  (reduceXTicks false))]
    (.. chart -xAxis (axisLabel "Weights") (tickFormat (.format js/d3 ",r")))
    (.. chart -yAxis (axisLabel "Count") (tickFormat (.format js/d3 ",r")))
    (.. js/d3 (select elem) (select "svg")
        (datum (clj->js hist-data))
        (call chart))))

(defn draw-histogram
  [hist owner]
  (let [elem   (om/get-node owner)
        data   (:data hist)
        labels (map #(/ (.round js/Math (* 100 %)) 100.0) (:labels hist))
        values (clj->js (make-data-series data labels))]
    (if (nil? data)
      (nv-clear-graph elem))
    (nv-bar-graph elem [#js {:key "Weights" :values values}])))

(defn layer-weight-histogram-view
  "Om component that displays the weight histogram of a given network layer"
  [weight-hist owner]
  (reify
    om/IRender
    (render [this]
      (dom/div #js {:className "layer-view panel"}
        (dom/svg nil nil)))
    om/IDidUpdate
    (did-update [this prev-props prev-state]
      (draw-histogram weight-hist owner))
    om/IDidMount
    (did-mount [this]
      (draw-histogram weight-hist owner))))

(defmethod net-perspective-view
  :weights
  [[_ app] owner]
  (om/component
    (let [net     (selected-net app)
          net-key (-> app :sel :net)
          layers  (rest (-> net :arch :layers))
          weights (:weights net)
          weight-hist     (:weight-hist net)
          weights-updated (:weights-updated net)]
      (if (empty? weights)
        (dom/div #js {:className "net-view"}
          (refresh-net-weights! app net-key)
          (om/build loading-icon nil))
        (apply dom/div #js {:className "net-view"}
               (concat [(dom/h4 nil "Network Weights")
                        (dom/button
                          #js {:className "refresh-weights-btn"
                               :disabled (if weights-updated false "disabled")
                               :onClick (fn [_] (refresh-net-weights! app net-key))}
                          "Refresh")]
                       (om/build-all layer-weights-view (map vector weights layers))
                       (om/build-all layer-weight-histogram-view weight-hist)))))))

(defn nv-line-graph
  "Draw a graph of Training & Validation errors using NVD3"
  [elem error-data]
  (let [chart (.. js/nv -models lineChart
                  (margin #js {:left 120})
                  (useInteractiveGuideline true)
                  (transitionDuration 350)
                  (showLegend true)
                  (showYAxis true)
                  (showXAxis true))]
    (.. chart -xAxis (axisLabel "Epochs") (tickFormat (.format js/d3 ",r")))
    (.. chart -yAxis (axisLabel "Error") (tickFormat (.format js/d3 ",r")))
    (.. js/d3 (select elem) (select "svg")
        (datum (clj->js error-data))
        (call chart))))

(def graph-colors ["#ff7f0e" "#2ca02c"])

(defn error-graph-data
  [tr-err val-err]
  (let [errorkinds (keys tr-err)]
    (flatten (map #(vector
      #js {:values (clj->js (make-data-series (%1 tr-err)))
           :key (str "Training " (name %1))
           :color (nth graph-colors %2)}
      #js {:values (clj->js (make-data-series (%1 val-err)))
           :key (str "Validation " (name %1))
           :color (nth graph-colors %2)
           :dash "3,5"}) errorkinds (range)))))

(defn draw-error-graph
  [stats owner]
  (let [elem    (om/get-node owner)
        tr-err  (:tr-err stats)
        val-err (:val-err stats)]
    (if (= 0 (-> tr-err first second count))
      (nv-clear-graph elem))
    (nv-line-graph elem (error-graph-data tr-err val-err))))

(defn error-graph
  "Om component that displays an error graph for the neural network"
  [stats owner]
  (reify
    om/IRender
    (render [this]
      (dom/div #js {:id "error-graph"}
        (dom/svg nil nil)))
    om/IDidUpdate
    (did-update [this prev-props prev-state]
      (draw-error-graph stats owner))
    om/IDidMount
    (did-mount [this]
      (draw-error-graph stats owner))))

(defn start-training!
  [app nepochs]
  (let [sn    (-> @app :sel :net)
        net   (-> @app :nets sn)
        trset (or (-> net :training :params :trset)
                  (first (first (filter #(compatible? net (second %))
                                        (:trsets @app)))))]
    (chsk-send! [:synaptic/start-training [sn trset nepochs]])
    (om/transact! app [:nets sn :training :state :state] (fn [_] :training))))

(defn stop-training!
  [app]
  (let [sn (-> @app :sel :net)]
    (chsk-send! [:synaptic/stop-training [sn]])
    (om/transact! app [:nets sn :training :state :state] (fn [_] :stopping))))

(defn net-training-state
  [training]
  (case (-> training :state :state)
    :training "training"
    :stopping "stopping"
    nil       "idle"))

(defn update-net-trset!
  [e app net update-layers?]
  (let [trset-key (keyword (.. e -target -value))]
    (om/transact! net [:training :params :trset] (fn [_] trset-key))
    (if update-layers?
      (let [layers  (-> net :arch :layers)
            trset   (-> @app :trsets trset-key)
            insize  (trset-fsize trset)
            outsize (-> trset :header :labels count)]
        (om/transact! (first layers) [:fieldsize] (fn [_] insize))
        (om/transact! (last layers)  [:n]         (fn [_] outsize))))))

(defn edit-net-trset
  [app owner {:keys [update-layers?]}]
  (reify
    om/IRender
    (render [this]
      (let [net (selected-net app)
            nets (:nets app)
            compat-trsets (filter #(compatible? net (second %)) (:trsets app))
            trset (or (-> net :training :params :trset)
                      (first (first compat-trsets))
                      (first (keys (:trsets app))))]
        (apply dom/select
               #js {:className "training-set-input"
                    :value (name trset)
                    :onChange #(update-net-trset! % app net update-layers?)}
          (map #(dom/option #js {:value (name (first %))} (name (first %)))
               (if update-layers? (:trsets app) compat-trsets)))))))

(defn training-state-view
  [app owner]
  (reify
    om/IInitState
    (init-state [this]
      {:nepochs 10})
    om/IRenderState
    (render-state [this {:keys [nepochs]}]
      (let [training (:training (selected-net app))]
        (dom/div #js {:className "tr-state-view panel"}
          (dom/p nil (str "Network has been trained for "
                          (or (-> training :stats :epochs) "0") " epochs."))
          (dom/p nil (str "Network is " (net-training-state training) "."))
          (dom/div nil
            (if (nil? (-> training :state :state))
              (dom/button #js {:className "start-training-btn"
                               :onClick (fn [_] (start-training! app nepochs))}
                          "Train")
              (dom/button #js {:className "start-training-btn"
                               :disabled "disabled"} "Train"))
            (dom/span nil " for ")
            (dom/input #js {:type "text" :className "edit-net-form-short-label"
                            :onChange
                            (fn [e]
                              (let [value (js/parseInt (.. e -target -value))]
                                (om/set-state! owner :nepochs value)))
                            :value nepochs} "")
            (dom/span nil " epochs, using training set: ")
            (om/build edit-net-trset app {:opts {:update-layers? false}}))
          (if (= (-> training :state :state) :training)
            (dom/button #js {:className "stop-training-btn"
                             :onClick (fn [_] (stop-training! app))} "Stop")
            (dom/button #js {:className "stop-training-btn"
                             :disabled "disabled"} "Stop")))))))

(defn training-param-regularization-view
  [training owner]
  (om/component
    (let [reg (-> training :params :regularization)]
      (dom/div #js {:style (display reg)}
        (if reg
          (dom/p nil (str "Regularization: " (-> reg :kind name (.toUpperCase))
                          " lambda: " (:lambda reg))))))))

(defn training-param-lrate-view
  [training owner]
  (om/component
    (let [lrate (-> training :params :learning-rate)
          a     (-> training :algo)]
      (dom/div #js {:style (display (or (= :backprop a) (= :rmsprop a)))}
        (dom/p nil (str "Learning rate: " (:epsilon lrate)))
        (dom/p #js {:style (display (:adaptive lrate))}
               (str "adaptive (min g: " (:ming lrate)
                    ", max g: " (:maxg lrate) ")"))))))

(defn training-param-momentum-view
  [training owner]
  (om/component
    (let [momentum (-> training :params :momentum)
          a        (-> training :algo)]
      (dom/div #js {:style (display momentum)}
        (dom/p nil (str "Momentum" (if (:nesterov momentum) " (Nesterov): " ": ")
                        "alpha: " (:alpha momentum)
                        (if (:alpha-start momentum)
                          (str " - progressive (alpha start: " (:alpha-start momentum)
                               ", alpha step:" (:alpha-step momentum) ")")
                          "")))))))

(defn training-param-view
  "Om component that displays training parameters of a neural net"
  [training owner]
  (om/component
    (dom/div #js {:className "tr-param-view panel"}
      (dom/p #js {:style #js {:fontWeight "bold"}} "Training Parameters")
      (dom/p nil (str "Error function: "
                      (if training (-> training :params :error-fn name) "")))
      (om/build training-param-regularization-view training)
      (dom/p nil (str "Algorithm: "
                      (if training (-> training :algo name) "")))
      (om/build training-param-lrate-view training)
      (om/build training-param-momentum-view training))))

(defn training-stats-view
  "Om component that displays training statistics of a neural net"
  [training owner]
  (om/component
    (dom/div #js {:className "tr-stats-view panel"}
      (dom/p #js {:style #js {:fontWEight "bold"}} "Training statistics")
      (om/build error-graph (:stats training)))))

(defmethod net-perspective-view
  :training
  [[_ app] owner]
  (om/component
    (let [training (:training (selected-net app))]
      (dom/div #js {:className "net-view"}
        (dom/h4 nil "Network Training")
        (om/build training-state-view app)
        (om/build training-param-view training)
        (om/build training-stats-view training)))))

(defn save-net!
  [net-key]
  (chsk-send! [:synaptic/save-net net-key]))

(defn net-view
  "Om component that displays the selected neural network"
  [app owner]
  (reify
    om/IInitState
    (init-state [this]
      {:ch (chan)
       :perspective :arch})
    om/IWillMount
    (will-mount [this]
      (go (loop []
            (when-let [[event arg] (<! (om/get-state owner :ch))]
              (if (= event :perspective) (om/set-state! owner :perspective arg))
              (recur)))))
    om/IRenderState
    (render-state [this {:keys [perspective ch]}]
      (let [sel (-> app :sel :net)]
        (dom/div #js {:className "net-panel panel" :style (display sel)}
          (if-let [net (selected-net app)]
            (dom/div nil
              (dom/h3 nil (-> net :header :name))
              (om/build net-navigation-bar net {:state {:perspective perspective}
                                                :opts {:ch ch}})
              (if (and (nil? (-> net :training :state :state))
                       (not= :saved (-> net :save-state)))
                (dom/button #js {:className "save-net-btn"
                                 :onClick (fn [_] (save-net! sel))} "Save")
                (dom/button #js {:className "save-net-btn"
                                 :disabled "disabled"} "Save"))
              (om/build net-perspective-view [perspective app]))
            (om/build loading-icon nil)))))))

; Network edition

(defn update-net-name!
  [e nets]
  (let [value (.. e -target -value)]
    (om/transact! nets [:new :header :name] (fn [_] value))))

(defn edit-net-header
  [nets owner]
  (om/component
    (dom/div nil
      (dom/input #js {:className "net-name-input"
                      :type "text" :value (-> nets :new :header :name)
                      :onChange #(update-net-name! % nets)}))))

(defn layer-name
  [i n]
  (cond
    (= i 0)       "Input layer"
    (= i (dec n)) "Output layer"
    :else         (str "Hidden layer #" i)))

(defn add-hidden-layer!
  [layers]
  (om/transact! layers []
                (fn [old-layers]
                  (vec (concat (butlast old-layers)
                               [{:type :fully-connected :n 0 :act-fn :sigmoid}
                                (last old-layers)])))))

(defn remove-hidden-layer!
  [layers i]
  (om/transact! layers []
                (fn [old-layers]
                  (vec (concat (subvec old-layers 0 i)
                               (subvec old-layers (inc i)))))))

(defn update-layer-type!
  [e layer]
  (let [value (keyword (.. e -target -value))]
    (om/transact! layer []
                  (fn [old-layer]
                    (merge {:type value :act-fn (:act-fn old-layer)}
                           (if (= :fully-connected value)
                             {:n 0}
                             {:feature-map {:k 1 :size [0 0]}}))))))

(defn update-layer-actfn!
  [e layer]
  (let [value (keyword (.. e -target -value))]
    (om/transact! layer [:act-fn] (fn [_] value))))

(defn update-enable-pooling!
  [e layer]
  (let [value (.. e -target -checked)]
    (om/transact! layer [:pool] (fn [_] (if value {:kind :max :size [0 0]})))))

(defn update-layer-units!
  [e layer path]
  (let [value (js/parseInt (.. e -target -value))]
    (om/transact! layer path (fn [_] (max value 0)))))

(defn edit-fieldsize-view
  [layer owner]
  (om/component
    (let [[k w h] (:fieldsize layer)]
      (dom/div #js {:className "edit-net-form-medium-label edit-net-layer-elem"}
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> k 0) (str k) "")
                        :onChange #(update-layer-units! % layer [:fieldsize 0])})
        (dom/span nil " x ")
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> w 0) (str w) "")
                        :onChange #(update-layer-units! % layer [:fieldsize 1])})
        (dom/span nil " x ")
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> h 0) (str h) "")
                        :onChange #(update-layer-units! % layer [:fieldsize 2])})))))

(defmulti edit-layer-units-view
  (fn [layer _ _] (:type layer)))

(defmethod edit-layer-units-view
  :input
  [layer owner {:keys [editable?]}]
  (om/component
    (if editable?
      (om/build edit-fieldsize-view layer)
      (dom/span #js {:className "edit-net-form-medium-label"}
                (layer-nunits-as-str layer)))))

(defmethod edit-layer-units-view
  :convolution
  [layer owner _]
  (om/component
    (let [fmap  (:feature-map layer)
          k     (:k fmap)
          [w h] (:size fmap)
          pool  (:pool layer)]
      (dom/div #js {:className "edit-net-form-medium-label edit-net-layer-elem"}
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> k 0) (str k) "")
                        :onChange #(update-layer-units!
                                     % layer [:feature-map :k])})
        (dom/span nil " x ")
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> w 0) (str w) "")
                        :onChange #(update-layer-units!
                                     % layer [:feature-map :size 0])})
        (dom/span nil " x ")
        (dom/input #js {:className "layer-unit-el-input" :type "text"
                        :value (if (> h 0) (str h) "")
                        :onChange #(update-layer-units!
                                     % layer [:feature-map :size 1])})
        (dom/br nil)
        (dom/span nil "pool")
        (dom/input #js {:type "checkbox" :checked (if pool "checked" false)
                        :onChange #(update-enable-pooling! % layer)})
        (if-let [[pw ph] (:size pool)]
          (dom/span nil
            (dom/input #js {:className "layer-unit-el-input" :type "text"
                            :value (if (> pw 0) (str pw) "")
                            :onChange #(update-layer-units!
                                         % layer [:pool :size 0])})
            (dom/span nil " x ")
            (dom/input #js {:className "layer-unit-el-input" :type "text"
                            :value (if (> ph 0) (str ph) "")
                            :onChange #(update-layer-units!
                                         % layer [:pool :size 1])})))))))

(defmethod edit-layer-units-view
  :fully-connected
  [layer owner {:keys [editable?]}]
  (om/component
    (let [nunits (:n layer)]
      (dom/div #js {:className "edit-net-form-medium-label edit-net-layer-elem"}
        (if editable?
          (dom/input #js {:className "layer-unit-input" :type "text"
                          :value (if (> nunits 0) (str nunits) "")
                          :onChange #(update-layer-units! % layer [:n])})
          (dom/input #js {:className "layer-unit-input" :type "text"
                          :value (if (> nunits 0) (str nunits) "")
                          :disabled "disabled"}))
        (dom/span nil " units")))))

(defn edit-layer-view
  [nets owner {:keys [i]}]
  (om/component
    (let [algo   (-> nets :new :training :algo)
          layers (-> nets :new :arch :layers)
          n      (count layers)
          layer  (nth layers i)
          ltype  (:type layer)
          act-fn (:act-fn layer)
          edit-nunits? (or (< 0 i (dec n))
                           (nil? (-> nets :new :training :params :trset)))]
      (dom/div nil
        (if (and (not= :perceptron algo) (= i (dec n)))
          (dom/div nil
            (dom/button #js {:className "edit-net-layer-elem"
                             :onClick (fn [_] (add-hidden-layer! layers))} "+")
            (dom/span nil " Add hidden layer")))
        (dom/div nil
          (if (< 0 i (dec n))
            (dom/button #js {:className "net-layer-margin edit-net-layer-elem"
                             :onClick (fn [_] (remove-hidden-layer! layers i))} "-")
            (dom/span #js {:className "net-layer-margin edit-net-layer-elem"} ""))
          (dom/span #js {:className "edit-net-form-label edit-net-layer-elem"}
                    (str (layer-name i n) ":"))
          (if (< 0 i (dec n))
            (apply dom/select #js {:className "layer-type-input edit-net-layer-elem"
                                   :value (name ltype)
                                   :onChange #(update-layer-type! % layer)}
              (map #(dom/option #js {:value (name %)} (name %)) layer-types))
            (dom/span #js {:className "layer-type-input edit-net-layer-elem"} ""))
          (om/build edit-layer-units-view layer {:opts {:editable? edit-nunits?}})
          (if (< 0 i)
            (apply dom/select #js {:className "layer-actfn-input edit-net-layer-elem"
                                   :value (name act-fn)
                                   :onChange #(update-layer-actfn! % layer)}
              (map #(dom/option #js {:value (name %)} (name %))
                   activation-functions))))))))

(defn edit-net-arch-view
  [nets owner]
  (om/component
    (let [layers (-> nets :new :arch :layers)]
      (apply dom/div #js {:className "edit-arch-view panel"}
        (cons (dom/p #js {:style #js {:fontWeight "bold"}} "Architecture")
              (map #(om/build edit-layer-view nets {:opts {:i %}})
                   (range (count layers))))))))

(defn update-training-algo!
  [e nets]
  (let [value (keyword (.. e -target -value))
        trset (-> @nets :new :training :params :trset)
        reg   (-> @nets :new :training :params :regularization)]
    (om/transact! nets [:new :training]
                  (fn [_] { :algo value
                            :params (assoc (default-training-params value)
                                           :trset trset :regularization reg) }))))

(defn edit-training-algo-view
  [nets owner]
  (om/component
    (dom/div nil
      (dom/span #js {:className "edit-net-form-medium-label"} "Algorithm:")
      (apply dom/select
             #js {:className "training-algo-input"
                  :value (name (-> nets :new :training :algo))
                  :onChange #(update-training-algo! % nets)}
        (map #(dom/option #js {:value (name %)} (name %))
             training-algorithms)))))

(defn update-training-errorfn!
  [e nets]
  (let [value (keyword (.. e -target -value))]
    (om/transact! nets [:new :training :params :error-fn] (fn [_] value))))

(defn edit-training-errorfn-view
  [nets owner]
  (om/component
    (let [algo     (-> nets :new :training :algo)
          error-fn (-> nets :new :training :params :error-fn)]
      (dom/div nil
        (dom/span #js {:className "edit-net-form-medium-label"} "Error function:")
        (apply dom/select
               #js {:className "training-algo-input"
                    :value (name error-fn)
                    :onChange #(update-training-errorfn! % nets)}
          (map #(dom/option #js {:value (name %)} (name %))
               (if (= :perceptron algo)
                 [:misclassification]
                 [:sum-of-squares :cross-entropy])))))))

(defn update-enable-regularization!
  [e nets]
  (let [value (.. e -target -checked)]
    (om/transact! nets [:new :training :params :regularization]
                  (fn [_] (if value { :lambda 0.1 :kind :l2 })))))

(defn update-regularization-kind!
  [e nets]
  (if-let [value (.. e -target -checked)]
    (om/transact! nets [:new :training :params :regularization :kind]
                  (fn [_] (keyword (.. e -target -value))))))

(defn update-regularization-lambda!
  [e nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :regularization :lambda]
                  (fn [_] (if (< 0 value) value 0.1)))))

(defn edit-training-regularization-view
  [nets owner]
  (om/component
    (let [training (-> nets :new :training)
          reg      (-> training :params :regularization)]
      (dom/div nil
        (dom/div nil
          (dom/span #js {:className "edit-net-form-medium-label"} "Regularization:")
          (dom/input #js {:type "checkbox" :checked (if reg "checked" false)
                          :onChange #(update-enable-regularization! % nets)}))
        (dom/div #js {:style (display reg)}
          (dom/input #js {:type "radio" :name "regularization-kind" :value "l1"
                          :checked (if (= :l1 (:kind reg)) "checked" false)
                          :onChange #(update-regularization-kind! % nets)})
          (dom/span #js {:className "edit-net-form-short-label"} "L1")
          (dom/input #js {:type "radio" :name "regularization-kind" :value "l2"
                          :checked (if (= :l2 (:kind reg)) "checked" false)
                          :onChange #(update-regularization-kind! % nets)})
          (dom/span #js {:className "edit-net-form-short-label"} "L2"))
        (dom/div #js {:style (display reg)}
          (dom/span #js {:className "edit-net-form-short-label"} "lambda:")
          (dom/input #js {:className "regularization-lambda-input" :type "text"
                          :value (:lambda reg)
                          :onChange #(update-regularization-lambda! % nets)}))))))

(defn update-lrate-epsilon!
  [e nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :learning-rate :epsilon]
                  (fn [_] (if (< 0 value 1) value 0.01)))))

(defn update-lrate-adaptive!
  [e nets]
  (let [value (.. e -target -checked)]
    (om/transact! nets [:new :training :params :learning-rate]
                  (fn [old-value]
                    (merge { :epsilon (:epsilon old-value) }
                           (if value { :adaptive true :ming 0.01 :maxg 10.0 }))))))

(defn update-lrate-minmaxg!
  [e k nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :learning-rate k]
                  (fn [_] (if (< 0 value) value (if (= :ming k) 0.01 10.0))))))

(defn edit-training-lrate-view
  [nets owner]
  (om/component
    (let [training (-> nets :new :training)
          lrate    (-> training :params :learning-rate)
          a        (-> training :algo)]
      (dom/div #js {:style (display (or (= :backprop a) (= :rmsprop a)))}
        (dom/div nil
          (dom/span #js {:className "edit-net-form-medium-label"} "Learning rate:")
          (dom/input #js {:className "lrate-epsilon-input" :type "text"
                          :value (:epsilon lrate)
                          :onChange #(update-lrate-epsilon! % nets)}))
        (dom/div nil
          (dom/input (if (:adaptive lrate)
                       #js {:type "checkbox" :checked "checked"
                            :onChange #(update-lrate-adaptive! % nets)}
                       #js {:type "checkbox"
                            :onChange #(update-lrate-adaptive! % nets)}))
          (dom/span nil " adaptive")
          (dom/div #js {:className "adaptive-lrate-params"
                        :style (display (:adaptive lrate))}
            (dom/span #js {:className "edit-net-form-short-label"} "min g:")
            (dom/input #js {:className "lrate-minmaxg-input" :type "text"
                            :value (:ming lrate)
                            :onChange #(update-lrate-minmaxg! % :ming nets)})
            (dom/span #js {:className "edit-net-form-short-label"} "max g:")
            (dom/input #js {:className "lrate-minmaxg-input" :type "text"
                            :value (:maxg lrate)
                            :onChange #(update-lrate-minmaxg! % :maxg nets)})))))))

(defn update-enable-momentum!
  [e nets]
  (let [value (.. e -target -checked)]
    (om/transact! nets [:new :training :params :momentum]
                  (fn [_] (if value { :alpha 0.9 })))))

(defn update-momentum-alpha!
  [e nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :momentum :alpha]
                  (fn [_] (if (< 0 value 1) value 0.9)))))

(defn update-progr-momentum!
  [e k nets]
  (if (= k :enable)
    (let [value (.. e -target -checked)]
      (om/transact! nets [:new :training :params :momentum]
                    (fn [old-value]
                      (merge { :alpha (:alpha old-value) }
                           (if value { :alpha-start 0.5 :alpha-step 0.05 })))))
    (let [value (js/parseFloat (.. e -target -value))]
      (om/transact! nets [:new :training :params :momentum k]
                    (fn [_] (if (< 0 value) value (if (= k :start) 0.5 0.05)))))))

(defn update-nesterov-momentum!
  [e nets]
  (let [value (.. e -target -checked)]
    (om/transact! nets [:new :training :params :momentum]
                  (fn [old-value]
                    (if value
                      (assoc old-value :nesterov true)
                      (dissoc old-value :nesterov))))))

(defn edit-training-momentum-view
  [nets owner]
  (om/component
    (let [training (-> nets :new :training)
          momentum (-> training :params :momentum)
          a        (-> training :algo)]
      (dom/div #js {:style (display (or (= :backprop a) (= :rprop a) (= :rmsprop a)))}
        (dom/div nil
          (dom/span #js {:className "edit-net-form-medium-label"} "Momentum:")
          (dom/input (if momentum 
                       #js {:type "checkbox" :checked "checked"
                            :onChange #(update-enable-momentum! % nets)}
                       #js {:type "checkbox"
                            :onChange #(update-enable-momentum! % nets)}))
          (dom/span #js {:style (display momentum)
                         :className "edit-net-form-short-label"} "alpha:")
          (dom/input #js {:style (display momentum)
                          :className "momentum-alpha-input" :type "text"
                          :value (:alpha momentum)
                          :onChange #(update-momentum-alpha! % nets)}))
        (dom/div #js {:style (display momentum)}
          (dom/input (if (:alpha-start momentum)
                       #js {:type "checkbox" :checked "checked"
                            :onChange #(update-progr-momentum! % :enable nets)}
                       #js {:type "checkbox"
                            :onChange #(update-progr-momentum! % :enable nets)}))
          (dom/span nil " progressive")
          (dom/div #js {:className "progressive-momentum-params"
                        :style (display (:alpha-start momentum))}
            (dom/span #js {:className "edit-net-form-short-label"} "start:")
            (dom/input #js {:className "momentum-startstep-input" :type "text"
                            :value (:alpha-start momentum)
                            :onChange #(update-progr-momentum! % :start nets)})
            (dom/span #js {:className "edit-net-form-short-label"} "step:")
            (dom/input #js {:className "momentum-startstep-input" :type "text"
                            :value (:alpha-step momentum)
                            :onChange #(update-progr-momentum! % :step nets)})))
        (dom/div #js {:style (display momentum)}
          (dom/input (if (:nesterov momentum)
                       #js {:type "checkbox" :checked "checked"
                            :onChange #(update-nesterov-momentum! % nets)}
                       #js {:type "checkbox"
                            :onChange #(update-nesterov-momentum! % nets)}))
          (dom/span nil " Nesterov"))))))

(defn update-rprop-minmaxs!
  [e k nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :rprop k]
                  (fn [_] (if (< 0 value) value (if (= :mins k) 1e-6 50.0))))))

(defn edit-training-rprop-view
  [nets owner]
  (om/component
    (let [training (-> nets :new :training)
          rprop    (-> training :params :rprop)]
      (dom/div #js {:style (display (= :rprop (:algo training)))}
        (dom/div nil
          (dom/span #js {:className "edit-net-form-short-label"} "min s:")
          (dom/input #js {:className "rprop-minmaxs-input" :type "text"
                          :value (:mins rprop)
                          :onChange #(update-rprop-minmaxs! % :mins nets)})
          (dom/span #js {:className "edit-net-form-short-label"} "max s:")
          (dom/input #js {:className "rprop-minmaxs-input" :type "text"
                          :value (:maxs rprop)
                          :onChange #(update-rprop-minmaxs! % :maxs nets)}))))))

(defn update-rmsprop-alpha!
  [e nets]
  (let [value (js/parseFloat (.. e -target -value))]
    (om/transact! nets [:new :training :params :rmsprop :alpha]
                  (fn [_] (if (< 0 value) { :alpha 0.9 })))))

(defn edit-training-rmsprop-view
  [nets owner]
  (om/component
    (let [training (-> nets :new :training)
          rmsprop  (-> training :params :rmsprop)]
      (dom/div #js {:style (display (= :rmsprop (:algo training)))}
        (dom/div nil
          (dom/span #js {:className "edit-net-form-short-label"} "alpha:")
          (dom/input #js {:className "rmsprop-alpha-input" :type "text"
                          :value (:alpha rmsprop)
                          :onChange #(update-rmsprop-alpha! % nets)}))))))

(defn edit-net-training-view
  [nets owner]
  (om/component
    (dom/div #js {:className "edit-training-view panel"}
      (dom/p #js {:style #js {:fontWeight "bold"}} "Training")
      (om/build edit-training-errorfn-view nets)
      (om/build edit-training-regularization-view nets)
      (om/build edit-training-algo-view nets)
      (om/build edit-training-lrate-view nets)
      (om/build edit-training-momentum-view nets)
      (om/build edit-training-rprop-view nets)
      (om/build edit-training-rmsprop-view nets))))

(defmulti valid-layer?
  (fn [i layer] (:type layer)))

(defmethod valid-layer?
  :default
  [i layer]
  false)

(defmethod valid-layer?
  :input
  [i layer]
  (and (= 0 i)
       (nil? (:act-fn layer))
       (nil? (:feature-map layer))
       (every? #(< 0 %) (:fieldsize layer))))

(defmethod valid-layer?
  :fully-connected
  [i layer]
  (and (< 0 i)
       ((:act-fn layer) activation-functions)
       (< 0 (:n layer))))

(defmethod valid-layer?
  :convolution
  [i layer]
  (and (< 0 i)
       ((:act-fn layer) activation-functions)
       (if-let [fmap (:feature-map layer)]
         (every? #(< 0 %) (into [(:k fmap)] (:size fmap))))
       (if-let [pool (:pool layer)]
         (every? #(< 0 %) (:size pool))
         true)))

(defn valid-net?
  [net]
  (let [layers (-> net :arch :layers)
        algo   (-> net :training :algo)]
    (and
      (every? true? (map valid-layer? (range) layers))
      (algo training-algorithms))))

(defn reset-net!
  [app]
  (om/transact! app [:nets :new] (fn [_] (default-net :backprop)))
  (init-new-net-trset! app))

(defn unique-key
  [nets net-name]
  (let [name-prefix (.replace net-name " " "-")]
    (loop [net-key (keyword name-prefix) k 1]
      (if (net-key (into #{} (keys nets)))
        (recur (keyword (str name-prefix k)) (inc k))
        net-key))))

(defn create-net!
  [sel app]
  (let [nets    (:nets @app)
        net     (:new nets)
        net-key (unique-key nets (-> net :header :name))]
    (om/transact! app [:nets :new :header :name] (fn [_] (name net-key)))
    (chsk-send! [:synaptic/create-net [net-key (-> @app :nets :new)]])
    (reset-net! app)
    (select-net! sel net-key)))

(defn edit-net-view
  [app owner]
  (om/component
    (let [sel  (:sel app)
          nets (:nets app)]
      (dom/div #js {:className "net-panel panel" :style (display (:new nets))}
        (om/build edit-net-header nets)
        (om/build edit-net-trset app {:opts {:update-layers? true}})
        (om/build edit-net-arch-view nets)
        (om/build edit-net-training-view nets)
        (dom/div #js {:style #js {:width "100%" :height "24px"}}
          (dom/button #js {:onClick (fn [_] (reset-net! app))} "Reset")
          (dom/button (if (valid-net? (:new nets))
                        #js {:onClick (fn [_] (create-net! sel app))}
                        #js {:disabled "disabled"})
                      "Save"))))))

(defmulti main-panel
  "Om component that displays either nets or sets"
  (fn [[view _] _] view))

(defmethod main-panel
  :nets
  [[_ app] owner]
  (om/component
    (let [sel  (:sel app)
          nets (:nets app)]
      (dom/div nil
        (om/build net-list-view [sel nets])
        (if (= :new (:net sel))
          (om/build edit-net-view app)
          (om/build net-view app))))))

; Training sets

(defn select-set!
  [sel set-key]
  (om/transact! sel [:set] (fn [_] {:set-key set-key :chunk {:type nil}})))

(defn set-list-view
  "Om component that displays the list of training sets, and allow selecting
  a set to be displayed"
  [[sel trsets] owner]
  (om/component
    (dom/div #js {:className "set-list panel"}
      (apply dom/ul nil
        (map #(dom/li #js {:onClick (fn [_] (select-set! sel %))
                           :className (if (= (-> sel :set :set-key) %) "selected" "")}
                      (or (-> trsets % :header :name) (name %)))
             (keys trsets))))))

(defn set-header-view
  "Om component that displays the header (summary) of a training set"
  [header owner]
  (om/component
    (let [batches      (map #(reduce + (vals %)) (-> header :batches))
          smpperbatch  (apply max batches)
          sumbatches   (reduce + batches)
          sumvalid     (if (:valid header) (reduce + (-> header :valid vals)) 0)
          totalsamples (+ sumbatches sumvalid)]
      (dom/div nil
        (dom/h3 nil (-> header :name))
        (dom/ul nil
          (dom/li nil (str totalsamples " samples drawn from "
                           (-> header :labels count) " classes."))
          (dom/li nil (str "Training set: "
                           (-> header :batches count) " batches of (max) "
                           smpperbatch " samples each ("
                           sumbatches " samples in total)."))
          (if (< 0 sumvalid)
            (dom/li nil (str "Cross-validation set: " sumvalid " samples."))))))))

(defn select-chunk!
  [sel-set set-chunk]
  (om/transact! sel-set [:chunk] (fn [_] set-chunk)))

(defn set-dataset-view
  [[chunk-type i data-set sel-set] owner]
  (om/component
    (let [n (reduce + (vals data-set))]
      (dom/div nil
        (dom/span nil (str (if (= :batch chunk-type)
                             (str "Batch #" i)
                             "Cross-validation set")
                           " (" n " samples)"))
        (apply dom/div nil
          (map #(dom/span #js {:className "trset-batch-chunk"
                               :onClick (fn [_]
                                          (select-chunk! sel-set
                                            {:type chunk-type :i i :start %}))}
                          (str %))
               (map (partial * chunk-size) (range (/ n chunk-size)))))))))

(defn set-datasets-view
  [[header sel-set] owner]
  (om/component
    (let [batches (:batches header)
          valid   (:valid header)]
      (dom/div #js {:className "panel set-batches-panel"}
        (apply dom/div nil
          (map #(om/build set-dataset-view [:batch (inc %) (nth batches %) sel-set])
               (range (count batches))))
        (if valid (om/build set-dataset-view [:valid 0 valid sel-set]))))))

(defn fetch-chunk!
  [set-key chunk-type i chunk-start]
  (chsk-send! [:synaptic/fetch-trset-chunk [set-key chunk-type i chunk-start]]))

(defn init-batches
  [old-batches app set-key]
  (or old-batches
      (vec (repeat (-> @app :trsets set-key :header :batches count) {}))))

(defn set-smp-view
  [[smp lb est [w h]] owner]
  (om/component
    (if (or lb est)
      (dom/div #js {:className (str "set-sample"
                                    (if (and lb est (not= lb est))
                                      " misclassified" ""))}
        (om/build neuron-view [smp [w h]])
        (dom/span #js {:className "set-smp-label"}
                  (cond
                    (and lb est) (str lb ", " est)
                    lb           (str lb)
                    est          (str est))))
      (om/build neuron-view [smp [w h]]))))

(defn set-chunk-view
  [[set-key trset set-chunk est] owner]
  (om/component
    (let [chunk-type  (:type set-chunk)
          chunk-start (:start set-chunk)
          data-set    (if chunk-type (if (= :batch chunk-type)
                                       (nth (:batches trset) (dec (:i set-chunk)))
                                       (:valid trset)))
          chunk-est   (if (and est chunk-type)
                        (let [ds-est (if (= :batch chunk-type)
                                       (nth (:batches est) (dec (:i set-chunk)))
                                       (:valid est))]
                          (vec (take chunk-size (drop chunk-start ds-est)))))
          [smp lb]    (get data-set chunk-start)
          [w h]       (-> trset :header :fieldsize)]
      (if smp
        (dom/div #js {:className "panel set-chunk-panel"}
          (dom/span nil (str (if (= :batch chunk-type)
                               (str "samples in batch #" (:i set-chunk))
                               "samples in cross-validation set")
                             " starting at #" chunk-start))
          (apply dom/div nil
            (om/build-all set-smp-view
                          (map vector smp (or lb (repeat nil))
                                      (or chunk-est (repeat nil)) (repeat [w h])))))
        (dom/div #js {:className "panel set-chunk-panel"}
          (when chunk-type
            (fetch-chunk! set-key chunk-type
                          (if (= :batch chunk-type) (dec (:i set-chunk)))
                          chunk-start)
            (om/build loading-icon nil)))))))

(defn set-view
  "Om component that displays the selected set"
  [[app tset est] owner]
  (om/component
    (let [set-key   (:set-key tset)
          set-chunk (:chunk tset)
          trset     (if set-key (-> app :trsets set-key))
          header    (:header trset)]
      (dom/div #js {:className "testset-panel panel" :style (display set-key)}
        (if tset
          (dom/div nil
            (om/build set-header-view header)
            (om/build set-datasets-view [header tset])
            (om/build set-chunk-view [set-key trset set-chunk est]))
          (om/build loading-icon nil))))))

(defn trset-view
  "Om component that displays the selected training set"
  [app owner]
  (om/component
    (let [trset (-> app :sel :set)]
      (om/build set-view [app trset nil]))))

; Network testing

(defn testset-view
  "Om component that displays the selected test set"
  [[app est] owner]
  (om/component
    (let [testset (-> (selected-net app) :testing :testset)]
      (om/build set-view [app testset est]))))

(defn update-net-testset!
  [e net]
  (let [testset-key (keyword (.. e -target -value))]
    (om/transact! net [:testing]
                  (fn [_] {:testset {:set-key testset-key :chunk {:type nil}}
                           :result nil}))))

(defn init-testset!
  [net compat-trsets]
  (when-let [testset-key (first (first compat-trsets))]
    (om/transact! net [:testing]
                  (fn [_] {:testset {:set-key testset-key :chunk {:type nil}}
                           :result nil}))
    testset-key))

(defn test-net!
  [net-key testset]
  (chsk-send! [:synaptic/test-net [net-key testset]]))

(defn select-net-testset
  [app owner]
  (om/component
    (let [net-key (-> app :sel :net)
          net     (-> app :nets net-key)
          compat-trsets (filter #(compatible? net (second %)) (:trsets app))
          testing (:testing net)
          testset (or (-> testing :testset :set-key)
                      (init-testset! net compat-trsets))]
      (dom/div nil
        (dom/div nil
          (dom/span nil "Test network with: ")
          (apply dom/select
                 #js {:className "training-set-input"
                      :value (if testset (name testset))
                      :onChange #(update-net-testset! % net)}
                 (map #(dom/option #js {:value (name (first %))} (name (first %)))
                      compat-trsets)))
        (if testset
          (dom/button #js {:className "testing-btn"
                           :onClick (fn [_] (test-net! net-key testset))} "Test")
          (dom/button #js {:className "testing-btn"
                           :disabled "disabled"} "Test"))
        (if-let [est (:result testing)]
          (om/build testset-view [app est]))))))

(defmethod net-perspective-view
  :testing
  [[_ app] owner]
  (om/component
    (dom/div #js {:className "net-view"}
      (dom/h4 nil "Network Testing")
      (om/build select-net-testset app))))

; Main panel and app

(defmethod main-panel
  :sets
  [[_ app] owner]
  (om/component
    (let [sel    (:sel app)
          trsets (:trsets app)]
      (dom/div nil
        (om/build set-list-view [sel trsets])
        (om/build trset-view app)))))

(def main-views {:nets "Nets" :sets "Sets"})

(defn main-navigation-bar
  [app owner {:keys [ch]}]
  (reify
    om/IInitState
    (init-state [this]
      {:view :nets})
    om/IRenderState
    (render-state [this {:keys [view]}]
      (dom/div #js {:className "navig-bar"}
        (apply dom/ul nil
          (map #(dom/li #js {:onClick (fn [_] (put! ch [:view %]))
                             :className (if (= view %) "selected" "")}
                        (main-views %))
               (keys main-views)))))))

(defn main-view
  "Om component that displays the main view of the application"
  [app owner]
  (reify
    om/IInitState
    (init-state [this]
      {:ch (chan) :view :nets})
    om/IWillMount
    (will-mount [this]
      (go (loop []
            (when-let [[event arg] (<! (om/get-state owner :ch))]
              (if (= event :view) (om/set-state! owner :view arg))
              (recur)))))
    om/IRenderState
    (render-state [this {:keys [view ch]}]
      (dom/div nil
        (dom/h2 nil "Synaptic Lab")
        (om/build main-navigation-bar app {:state {:view view} :opts {:ch ch}})
        (dom/div #js {:style #js {:width "100%" :height "24px"}}
          (dom/button #js {:onClick (fn [_] (refresh-state! app))} "Refresh"))
        (om/build main-panel [view app])))))

(defmulti handle-event
  "Handle events based on the event ID"
  (fn [[ev-id ev-arg] app owner] ev-id))

(defmethod handle-event
  :synaptic/update-state
  [[_ [path new-state]] app owner]
  (om/transact! app path
                (fn [old-state]
                  (if (map? old-state) (merge old-state new-state) new-state)))
  (if (= (first path) :trsets)
    (init-new-net-trset! app)))

(defmethod handle-event
  :synaptic/update-trset-chunk
  [[_ [set-key chunk-type i chunk-start chunk-smp-lb]] app owner]
  (if (= :batch chunk-type)
    (om/transact! app [:trsets set-key :batches]
                  (fn [old-batches]
                    (let [batches (init-batches old-batches app set-key)]
                      (assoc-in batches [i chunk-start] chunk-smp-lb))))
    (om/transact! app [:trsets set-key :valid]
                  (fn [old-valid]
                    (assoc (or old-valid {}) chunk-start chunk-smp-lb)))))

(defmethod handle-event
  :default
  [_ app owner]
  nil)

(defmethod handle-event
  :default
  [event app owner]
  (println "UNKNOWN EVENT: " event))

(defn event-loop
  "Handle inbound events"
  [app owner]
  (go (loop [[op arg] (:event (<! ch-chsk))]
        (println "-" op)
        (case op
          :chsk/recv (handle-event arg app owner)
          :chsk/state (if (:first-open? arg)
                        (do 
                          (println "Connection established")
                          (refresh-state! app))
                        (println "State changed to " arg))
          nil)
        (recur (:event (<! ch-chsk))))))

(defn application
  "Om component that represents our application, and handles events"
  [app owner]
  (reify
    om/IWillMount
    (will-mount [this]
      (event-loop app owner))
    om/IRender
    (render [this]
      (om/build main-view app))))

(om/root application app-state
  {:target (. js/document (getElementById "app"))})


