---
trigger: always_on
---

# AI Development Rules

1. **Keep files concise and maintainable.**

   * No file should exceed **700 lines of code**.
   * Split large implementations into logical modules.
   * Organize code using a clear folder and subfolder structure based on functionality and project requirements.

2. **Use meaningful file names.**

   * Every file name should clearly describe its purpose.
   * Keep names short, readable, and consistent with the project's naming convention.

3. **Write useful comments.**

   * Add comments only where they improve understanding.
   * Explain complex logic, algorithms, assumptions, and business rules.
   * Avoid obvious or redundant comments.

4. **Follow a consistent coding style.**

   * Maintain uniform formatting, indentation, and naming conventions throughout the project.
   * Use descriptive names for variables, functions, classes, and modules.

5. **Prioritize readability over cleverness.**

   * Write code that is easy to understand, maintain, and extend.
   * Prefer simple and clear solutions whenever possible.

6. **Ensure modularity and reusability.**

   * Avoid code duplication.
   * Create reusable functions, utilities, and components.

7. **Maintain proper separation of concerns.**

   * Each file, class, or module should have a single clear responsibility.
   * Business logic, UI, configuration, and data access should remain separated.

8. **Handle errors gracefully.**

   * Implement proper error handling and validation.
   * Provide meaningful error messages and logs.

9. **Keep documentation updated.**

   * Update relevant documentation whenever functionality changes.
   * Include setup instructions, architecture notes, and usage examples when necessary.

10. **Optimize only when necessary.**

    * Focus on correctness and maintainability first.
    * Optimize performance only after identifying actual bottlenecks.

11. **Maintain security best practices.**

    * Never hardcode secrets, API keys, passwords, or sensitive information.
    * Store secrets securely using environment variables or secret management systems.

12. **Write testable code.**

    * Design modules and functions to be easily testable.
    * Include unit and integration tests for critical functionality.

13. **Review dependencies carefully.**

    * Use only necessary libraries and frameworks.
    * Avoid adding dependencies without clear justification.

14. **Keep architecture scalable.**

    * Design solutions that can accommodate future growth and feature additions without major refactoring.

15. **Always explain major decisions.**

    * For significant architectural, design, or implementation choices, provide a brief rationale in comments or documentation.
