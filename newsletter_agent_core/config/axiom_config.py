"""
Axiom Configuration Loader

This module provides configuration management for the 10 European axioms
that guide structural analysis of newsletter items.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class AxiomConfig:
    """
    Configuration loader for axiom-based analysis.
    
    Loads and provides access to the 10 non-negotiable axioms used for
    structural sanity checking of newsletter items.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the axiom configuration loader.
        
        Args:
            config_path: Optional path to the configuration JSON file.
                        If not provided, uses the default location.
        """
        if config_path is None:
            # Default to the axiom_config.json in the same directory
            config_path = Path(__file__).parent / "axiom_config.json"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._axioms: List[Dict[str, Any]] = []
        self._reality_gates: List[Dict[str, Any]] = []
        self._filtering: Dict[str, Any] = {}
        
    def load(self) -> None:
        """
        Load the configuration from the JSON file.
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
            ValueError: If the configuration structure is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file: {e.msg}",
                e.doc,
                e.pos
            )
        
        # Validate required fields
        if "axioms" not in self._config:
            raise ValueError("Configuration must contain 'axioms' field")
        if "reality_gates" not in self._config:
            raise ValueError(
                "Configuration must contain 'reality_gates' field"
            )
        if "filtering" not in self._config:
            raise ValueError("Configuration must contain 'filtering' field")
        
        self._axioms = self._config["axioms"]
        self._reality_gates = self._config["reality_gates"]
        self._filtering = self._config["filtering"]
        
        # Validate axioms structure
        for axiom in self._axioms:
            required_fields = ["id", "title", "description", "keywords"]
            missing_fields = [f for f in required_fields if f not in axiom]
            if missing_fields:
                raise ValueError(
                    f"Axiom {axiom.get('id', 'unknown')} missing fields: "
                    f"{missing_fields}"
                )
    
    def get_axioms(self) -> List[Dict[str, Any]]:
        """
        Get all axioms.
        
        Returns:
            List of axiom dictionaries, each containing id, title, description,
            and keywords.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._axioms:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        return self._axioms.copy()
    
    def get_axiom_by_id(self, axiom_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific axiom by its ID.
        
        Args:
            axiom_id: The ID of the axiom to retrieve (1-10).
            
        Returns:
            The axiom dictionary if found, None otherwise.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._axioms:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        for axiom in self._axioms:
            if axiom["id"] == axiom_id:
                return axiom.copy()
        return None
    
    def get_prompt_text(self) -> str:
        """
        Generate formatted prompt text containing all axioms.
        
        Returns:
            A formatted string containing all 10 axioms, suitable for
            inclusion in LLM prompts.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._axioms:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        lines = ["## THE 10 AXIOMS (NON-NEGOTIABLE)\n"]
        for axiom in self._axioms:
            lines.append(
                f"{axiom['id']}. {axiom['title']}."
            )
        
        return "\n".join(lines)
    
    def get_reality_gates(self) -> List[Dict[str, Any]]:
        """
        Get all reality gate definitions.
        
        Returns:
            List of reality gate dictionaries.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._reality_gates:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        return self._reality_gates.copy()
    
    def get_max_violations(self, mode: str = "balanced") -> int:
        """
        Get the maximum allowed violations for a specific mode.
        
        Args:
            mode: The filtering mode ("strict", "balanced", or "exploratory").
                 Defaults to "balanced".
                 
        Returns:
            The maximum number of violations allowed for the specified mode.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
            ValueError: If the mode is not recognized.
        """
        if not self._filtering:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        modes = self._filtering.get("modes", {})
        if mode not in modes:
            raise ValueError(
                f"Unknown mode '{mode}'. Available modes: {list(modes.keys())}"
            )
        
        return modes[mode]["max_violations"]
    
    def get_version(self) -> str:
        """
        Get the configuration version.
        
        Returns:
            The version string.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._config:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        return self._config.get("version", "unknown")
    
    def get_metadata(self) -> Dict[str, str]:
        """
        Get configuration metadata (version, last_updated, language).
        
        Returns:
            Dictionary containing metadata fields.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._config:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        return {
            "version": self._config.get("version", "unknown"),
            "last_updated": self._config.get("last_updated", "unknown"),
            "language": self._config.get("language", "en")
        }
