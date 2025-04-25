from arango import ArangoClient
from pytrie import StringTrie
from rapidfuzz import fuzz
import re

# Define glossary with 15 terms, including multi-word phrases
GLOSSARY = {
    Airplane Parts: Components used in the construction and maintenance of aircraft.,
    Control Tower: A facility that manages aircraft takeoff, landing, and ground traffic.,
    Jet Engine: A type of engine that propels aircraft by expelling jet streams.,
    Flight Path: The planned route or trajectory an aircraft follows during flight.,
    Avionics System: Electronic systems used in aircraft for navigation and communication.,
    Landing Gear: The undercarriage of an aircraft used during takeoff and landing.,
    Aerodynamic Design: The shape of an aircraft optimized to reduce air resistance.,
    Cabin Pressure: The controlled air pressure inside an aircraft cabin for passenger comfort.,
    "Wing Flaps": "Movable parts on aircraft wings to control lift and drag.",
    "Radar System": "A system used to detect aircraft and other objects in the airspace.",
    "Fuel Tank": "A container for storing fuel on an aircraft.",
    "Cockpit Display": "The interface in the cockpit showing flight information.",
    "Navigation Aid": "Tools or systems that assist in aircraft navigation.",
    "Turbulence Alert": "A warning system for detecting unstable air conditions."
}

def find_glossary_terms(search_string, similarity_threshold=97):
    """
    Search for glossary terms (including multi-word phrases) in the provided string
    using pytrie and filter with rapidfuzz for whole keyword matches.
    
    Args:
        search_string (str): The text to search for glossary terms.
        similarity_threshold (float): Minimum rapidfuzz similarity score (default: 97).
    
    Returns:
        list: List of dictionaries containing matched terms and their definitions.
    """
    if not search_string:
        return []
        
    # Initialize StringTrie with glossary terms
    trie = StringTrie()
    for term in GLOSSARY:
        trie[term.lower()] = term

    # Convert search string to lowercase for case-insensitive matching
    search_string_lower = search_string.lower()

    # Find potential matches using pytrie and regex for whole keyword
    potential_matches = set()
    for term in GLOSSARY:
        term_lower = term.lower()
        # Create regex pattern based on whether its a multi-word term or single word