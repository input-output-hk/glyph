# Contributing

## What & How Can You Contribute?

1. **Feedback**

   Contributions in the form of feedback and issues are very much welcome. Whether it may be a suggestion, a bug report, or maybe some questions that you have. It helps in improving Aiken in over time and these are the best kind of contributions to start with.

   Do not hesitate to add _thumbs up :+1:_ on open issues you support to show your interest.

2. **Documentation**

   At this time we have not yet established standards for documentation of this project. Guidelines for documentation soon to follow.

3. **Code**

   CI runs the following commands. To save time, it's good to run these on your local machine before pushing to origin.
   ```
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo build
   cargo test
   ```

   **Changelog**

   Please add an entry into [CHANGELOG.md](./CHANGELOG.md) when submitting changes. New entries should go into the `[next] YYYY-MM-DD` section. This let's us keep track of unreleased changes
   for use in release notes.

   Once a release is ready `[next] YYYY-MM-DD` gets replaced with a version number and a the release date `[0.0.0] 2009-01-03`. Usually the maintainers will handle the section renaming along with creating a new _empty_ `[next] YYYY-MM-DD` section at the top of the changelog.

   Each release section will be further broken down into three sections named `Added`, `Changed`, and `Removed`.

   ```md
   ## [next] - YYYY-MM-DD

   ### Added

   - some new thing

   ### Changed

   - fixed or updated something

   ### Removed

   - something is gone now
   ```
