(defproject synaptic-lab "0.1.0-SNAPSHOT"
  :description "Visualise and experiment with neural networks in Clojure"
  :url "https://github.com/japonophile/synaptic-lab"
  :dependencies [[org.clojure/clojure "1.7.0-beta1"]
                 [org.clojure/clojurescript "0.0-3196"]
                 [org.clojure/core.async "0.1.346.0-17112a-alpha"]
                 [org.omcljs/om "0.8.8"]
                 [com.taoensso/sente "1.4.1"]
                 [ring/ring-core "1.3.2"]
                 [http-kit "2.1.19"]
                 [jetty/javax.servlet "5.1.12"]
                 [compojure "1.3.3"]
                 [synaptic "0.2.0"]
                 [clatrix/clatrix "0.4.0"]]
  :plugins [[lein-cljsbuild "1.0.5"]
            [lein-ring "0.9.3"]]
  :hooks [leiningen.cljsbuild]
  :source-paths ["src/clj"]
  :cljsbuild { 
    :builds {
      :main {
        :source-paths ["src/cljs"]
        :compiler {:output-to "resources/public/js/cljs.js"
                   :optimizations :simple
                   :pretty-print true}
        :jar true}}}
  :main synaptic-lab.server
  :jvm-opts ["-Xmx1536m"]
  :ring {:handler synaptic-lab.server/app}
  :repl-options {:timeout 300000}
  :keep-non-project-classes true)

