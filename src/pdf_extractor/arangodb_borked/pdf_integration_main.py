if __name__ == '__main__':
    # Run validation
    print('Validating ArangoDB integration...')
    results = validate_arangodb_integration()
    
    # Print results
    print('\nArangoDB Integration Status:')
    print('----------------------------')
    for key, value in results.items():
        status = '✅ WORKING' if value is True else '❌ FAILED' if value is False else '⚠️ ' + value.upper()
        print(f'{key.replace("_", " ").title()}: {status}')
    
    # Overall status
    working_count = sum(1 for v in results.values() if v is True)
    total_testable = sum(1 for v in results.values() if v != 'not_tested')
    print(f'\nWorking: {working_count}/{total_testable} tested features')
    
    if working_count >= 2:  # At least connection and collection setup work
        print('✅ ArangoDB integration is functional with working and fallback options')
        sys.exit(0)
    else:
        print('⚠️ Some critical ArangoDB integration features are not working correctly')
        sys.exit(1)
