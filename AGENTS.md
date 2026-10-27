# mikenavi fleet fork

This branch is the kyojin source. Read README.fleet.md and local/source.lock.json first.
The local features are committed source, not deployment-time patches. Preserve
feature regressions when updating upstream. Verify source identities and run
applicable CPU tests before publishing a new fleet tag; GPU/backend and workload
qualification are separate. Weights, host profiles, credentials and caches stay
outside this repository. Build/deploy exact commits, not a moving branch.
For Strata, the application components.lock.json selects the native engine commit;
the stock upstream installer/feed does not select the custom engine automatically.
