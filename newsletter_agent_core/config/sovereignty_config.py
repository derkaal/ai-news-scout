"""
Sovereignty Thesis Configuration Loader

This module provides configuration management for the European sovereignty theses
that guide content filtering in the newsletter agent.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class SovereigntyConfig:
    """
    Configuration loader for sovereignty theses.
    
    Loads and provides access to the 14 European sovereignty theses used for
    content filtering, along with filtering thresholds and modes.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the sovereignty configuration loader.
        
        Args:
            config_path: Optional path to the configuration JSON file.
                        If not provided, uses the default location.
        """
        if config_path is None:
            # Default to the sovereignty_theses.json in the same directory
            config_path = Path(__file__).parent / "sovereignty_theses.json"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._theses: List[Dict[str, Any]] = []
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
        if "theses" not in self._config:
            raise ValueError("Configuration must contain 'theses' field")
        if "filtering" not in self._config:
            raise ValueError("Configuration must contain 'filtering' field")
        
        self._theses = self._config["theses"]
        self._filtering = self._config["filtering"]
        
        # Validate theses structure
        for thesis in self._theses:
            required_fields = ["id", "title", "text", "category", "keywords"]
            missing_fields = [f for f in required_fields if f not in thesis]
            if missing_fields:
                raise ValueError(
                    f"Thesis {thesis.get('id', 'unknown')} missing fields: {missing_fields}"
                )
    
    def get_theses(self) -> List[Dict[str, Any]]:
        """
        Get all sovereignty theses.
        
        Returns:
            List of thesis dictionaries, each containing id, title, text,
            category, and keywords.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._theses:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        return self._theses.copy()
    
    def get_thesis_by_id(self, thesis_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific thesis by its ID.
        
        Args:
            thesis_id: The ID of the thesis to retrieve (1-14).
            
        Returns:
            The thesis dictionary if found, None otherwise.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._theses:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        for thesis in self._theses:
            if thesis["id"] == thesis_id:
                return thesis.copy()
        return None
    
    def get_prompt_text(self) -> str:
        """
        Generate formatted prompt text containing all theses.
        
        Returns:
            A formatted string containing all 14 theses, suitable for
            inclusion in LLM prompts.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._theses:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        lines = ["European Sovereignty Theses for Retail AI:\n"]
        for thesis in self._theses:
            lines.append(
                f"{thesis['id']}. **{thesis['title']}:** {thesis['text']}"
            )
        
        return "\n".join(lines)
    
    def get_threshold(self, mode: str = "balanced") -> float:
        """
        Get the filtering threshold for a specific mode.
        
        Args:
            mode: The filtering mode ("strict", "balanced", or "exploratory").
                 Defaults to "balanced".
                 
        Returns:
            The threshold value for the specified mode.
            
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
        
        return modes[mode]["threshold"]
    
    def get_min_aligned_theses(self) -> int:
        """
        Get the minimum number of aligned theses required for content to pass filtering.
        
        Returns:
            The minimum number of theses that must be aligned.
            
        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if not self._filtering:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        
        return self._filtering.get("min_aligned_theses", 1)
    
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