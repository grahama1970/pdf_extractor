# CODING PRINCIPLES
> Core development philosophy for the pdf_extractor project

## Functionality First Approach
- **95/5 Rule**: Use 95% functionality from existing packages with only 5% customization
- Selecting the right package is MORE important than writing clever new code
- Always prefer a functional approach over complex custom implementations
- Focus on making the code work correctly before making it elegant

## Minimalist Architecture
- Only use class architecture when absolutely necessary:
  1. When using Pydantic models for data validation
  2. When maintaining state (like Section Hierarchy)
  3. When implementing established design patterns
- Prefer simple functions and pure functional approaches
- Use composition over inheritance

## Prioritization of Concerns
1. **Working Code**: Functionality that produces expected results is the top priority
2. **Validation**: Comprehensive validation against expected outputs
3. **Clarity**: Code that is simple and easy to understand
4. **Static Analysis**: Address Pylance/typing errors only after the above are achieved

## Package Selection Guidelines
- Choose established packages with active maintenance
- Prefer packages with comprehensive documentation
- Select packages that handle edge cases gracefully
- Avoid packages with excessive dependencies
- Consider memory and performance implications

## Problem-solving Approach
1. Research existing solutions thoroughly
2. Start with the simplest possible implementation
3. Test against real PDF examples early and often
4. Incrementally improve while maintaining validation
5. Document any non-obvious decisions or techniques

## Anti-patterns to Avoid
- Premature optimization
- Overengineering and unnecessary abstraction
- Clever code that's difficult to understand
- Reinventing functionality that exists in established packages
- Focusing on style/linting issues before functionality is validated

## Up-to-date Package Research
- **NEVER** rely on outdated knowledge about packages
- **ALWAYS** research current package information before selection
- Document your research findings before making package decisions
- Test packages with small examples before full integration
- See CODE_DETAILS.md for specific package research requirements and procedures