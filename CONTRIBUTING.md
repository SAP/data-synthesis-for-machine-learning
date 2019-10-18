# Contributing to Data Synthesis for Machine Learning

You are welcome to contribute code to the Data Synthesis for Machine Learning in order to fix bugs or to implement new features.

You must be aware of the Apache License (which describes contributions) and **agree to the Contributors License Agreement**. This is common practice in major Open Source projects. To make this process as simple as possible, we are using *[CLA assistant](https://cla-assistant.io/)* for individual contributions. CLA assistant is an open source tool that integrates with GitHub very well and enables a one-click experience for accepting the CLA. For company contributors, special rules apply. See the respective section below for details.

## Contributor License Agreement
When you contribute code, documentation, or anything else, you have to be aware that your contribution is covered by the same [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0) that is applied to the UI5 Build and Development Tooling itself.

In particular, you need to agree to the Individual Contributor License Agreement, which can be [found here](https://gist.github.com/CLAassistant/bd1ea8ec8aa0357414e8). This applies to all contributors, including those contributing on behalf of a company.

If you agree to its content, you simply have to click on the link posted by the CLA assistant as a comment in the pull request. Click it to check the CLA, then accept it on the following screen if you agree to it. The CLA assistant saves this decision for upcoming contributions to that repository and notifies you, if there is any change to the CLA in the meantime.

### Company Contributors
If employees of a company contribute code, in **addition** to the individual agreement mentioned above, one company agreement must be submitted. This is mainly for the protection of the contributing employees.

A company representative authorized to do so needs to download, fill in, and print the [Corporate Contributor License Agreement](/docs/SAP%20Corporate%20Contributor%20License%20Agreement.pdf) form and then proceed with one of the following options:

- Scan and e-mail it to [opensource@sap.com](mailto:opensource@sap.com)
- Fax it to: +49 6227 78-45813
- Send it by traditional letter to: OSPO Core, Dietmar-Hopp-Allee 16, 69190 Walldorf, Germany
  
The form contains a list of employees who are authorized to contribute on behalf of your company. When this list changes, please let us know.

## How to report an issue
If you find a bug, you are welcome to report it. Just create a new issue in our repository and give the detailed description and reproduce steps.

## Issue handling process
When an issue is reported, we will look at it and either confirm it as a real issue (by giving the "in progress" label), close it if it is not an issue, or ask for more details. In-progress issues are then assigned to a committer in GitHub. An issue that is about a real bug is closed as soon as the fix is committed. 

## How to Contribute
1. Create a branch by forking the repository and apply your change.
1. Commit and push your change on that branch.  
   Commit Message Style: `<commit-type>: <commit-description>`   
   - commit-type: 
     + `fix` - a bug fix (note: this will indicate a release)
     + `feat` - a new feature (note: this will indicate a release)
     + `docs` - documentation only changes
     + `style` - changes that do not affect the meaning of the code
     + `refactor` - a code change that neither fixes a bug nor adds a feature
     + `perf` - a code change that improves performance
     + `test` - adding missing tests
     + `chore` - changes to the build process or auxiliary tools and libraries such as documentation generation
     + `revert` - revert to a commit
     + `WIP` - work in progress    
   - commit-description: detail information of the commit 
1. Create a pull request in the repository.
1. Follow the link posted by the CLA assistant to your pull request and accept it, as described above.
1. Wait for our code review and approval, possibly enhancing your change on request.
1. Once the change has been approved and merged, we will inform you in a comment.
