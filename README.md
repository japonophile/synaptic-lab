# Synaptic Lab

Synaptic Lab is a web-based user interface to create, train and test neural networks in an
intuitive way.

It uses the [Synaptic](https://github.com/japonophile/synaptic/) neural net Clojure library.
Synaptic Lab is written in Clojure and Clojurescript and uses core.async, Om and Sente.

# Usage

- Start the app:
```
$ lein run
Compiling ClojureScript.
mnist10k
Public root is /Users/antoine/git/github/synaptic-lab/resources/public
Starting Sente server on port 3000 ...
```

- Connect to http://localhost:3000/ and start creating and training your own neural nets!

The mnist10k data set is available in this repository so you can start experimenting
right away.  Of course, you can also drop your own data sets in the ./data directory
and they will be loaded when starting the app.

## License

Copyright © 2015 Antoine Choppin

Distributed under the Eclipse Public License, the same as Clojure.

