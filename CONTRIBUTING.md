# Contributing to `elastix` #

**Thank you for considering contributing to `elastix`!**


### Do you have questions about the source code? ###

* Ask any question about how to use `elastix` on the [mailing list](https://groups.google.com/forum/#!forum/elastix-imageregistration).

* Do not open an issue on GitHub for general questions, registration questions, or issues you may have while running `elastix`. _GitHub issues are primarily intended for bug reports and fixes._

* General information about `elastix` can be found on our [wiki](https://github.com/SuperElastix/elastix/wiki) or at the [website](https://elastix.lumc.nl/).

* [The manual](https://github.com/SuperElastix/elastix/releases/download/5.0.0/elastix-5.0.0-manual.pdf) explains much of the inner workings of image registration.

### Did you find a bug? ###

* _Ensure the bug was not already reported_ by searching on GitHub under [Issues](https://github.com/SuperElastix/elastix/issues).

* If you're unable to find an open issue addressing the problem, you can [open a new one](https://github.com/SuperElastix/elastix/issues/new). Be sure to include a _title and clear description_, as much relevant information as possible. A _code sample with a test_ demonstrating the expected behavior that is not occurring would be awesome.

### Do you intend to add a new feature or change an existing one? ###

* Suggest your change on the `elastix` mailing list.

* Do not open an issue on GitHub until you have collected positive feedback about the change.

### Did you write a patch that fixes a bug? ###

* Open a new [GitHub pull request](https://github.com/SuperElastix/elastix/pull/new/main) (PR) with the patch.

* Make sure the PR is done with respect to the [main branch](https://github.com/SuperElastix/elastix/tree/main).

* Ensure the PR description (log message) _clearly describes the problem and solution_. Include the relevant issue number if applicable. One-line messages are fine for small changes, but bigger changes should look like this:
    $ git commit -m "ENH: A brief summary of the commit
    >
    > A paragraph describing what changed and its impact."

* We use the following tags for commit messages:
  - ENH: for functional enhancements of the code
  - BUG: for fixing runtime issues (crash, segmentation fault, exception, or incorrect result)
  - COMP: for compilation issues (error or warning)
  - PERF: for performance improvements
  - STYLE: a change that does not impact the logic or execution of the code (improve coding style, comments, documentation)

The body of the message should clearly describe the motivation of the commit
(**what**, **why**, and **how**). In order to ease the task of reviewing
commits, the message body should follow the following guidelines:

  1. Leave a blank line between the subject and the body.
  This helps `git log` and `git rebase` work nicely, and allows to smooth
  generation of release notes.
  2. Try to keep the subject line below 73 characters.
  3. Capitalize the subject line.
  4. Do not end the subject line with a period.
  5. Use the imperative mood in the subject line (e.g. `BUG: Add missing spacing`).
  6. Use semantic line feeds to separate different ideas, which improves the
  readability.
  7. Be concise, but honor the change: if significant alternative solutions
  were available, explain why they were discarded.
  8. If the commit refers to a topic discussed in elastix' mailing list, or fixes
  a regression test, provide the link. If it fixes a compiler error, provide a
  minimal verbatim message of the compiler error. If the commit closes an
  issue, use the [GitHub issue closing
  keywords](https://help.github.com/en/articles/closing-issues-using-keywords).

Keep in mind that the significant time is invested in reviewing commits and
*pull requests*, so following these guidelines will greatly help the people
doing reviews.

These guidelines are largely inspired by Chris Beam's
[How to Write a Commit Message](https://chris.beams.io/posts/git-commit/)
post.


* Ensure the PR adheres to our [coding conventions](#coding-conventions), the PR code format will be checked with clangFormat on the CI. The hook can be used locally, for details see below.

* We will review your PR and possibly suggest changes. All PR will be tested on the GitHub Actions CI. When all lights are green, the PR will be merged.

* More information on pull requests can be found [here](https://help.github.com/articles/creating-a-pull-request/) and [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962).



<!--
### **Do you want to contribute to the `elastix` documentation?*

* Please read [Contributing to the Rails Documentation](http://edgeguides.rubyonrails.org/contributing_to_ruby_on_rails.html#contributing-to-the-rails-documentation).
-->
## Pre-commit clangFormat hook ##

A pre-commit hook that formats the code in the repository according to the correct clangFormat, is available for git.
In order to use the hook, it is has to be moved to the .git/hooks folder locally:

For Unix:

  `cp .githooks/pre-commit .git/hooks`

  `chmod u+x .git/hooks/pre-commit`

For Windows:

  `copy .githooks\pre-commit .git\hooks`

The pre-commit hooks is triggered by any git commit message, if files that don't adhere to the correct clangFormat style are found, these files are changes.
Run `git status` or `git diff` to see these changes.
Changes, if found, have to be re-added and re-commited with git, make sure to use the --amend flag when re-commiting to keep the commit history clean.

## Coding conventions ##

Start reading our code and you'll get the hang of it. We optimize for readability:

* We indent using two spaces (soft tabs). This ensures a consistent layout of the code on different text editors.

* We ALWAYS put spaces after list items and function arguments (`[1, 2, 3]`, not `[1,2,3]`), around operators (`x += 1`, not `x+=1`), etc.

* Use `/** ... */` style documentation.

* Member variables of classes should start their name with `m_`. For example: `m_NumberOfIterations`.

* Type definitions should start with a capital. ImageType for example, instead of imageType.

* This is open source software. Consider the people who will read your code, and make it look nice for them. It's sort of like driving a car: Perhaps you love doing donuts when you're alone, but with passengers the goal is to make the ride as smooth as possible.

## Code of conduct ##

Please respect our [code of conduct](CODE_OF_CONDUCT.md).

Thanks,
The `elastix` team
