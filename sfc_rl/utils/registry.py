"""Registry pattern for pluggable components."""

from typing import Dict, Type, TypeVar, Callable, Any, Optional
import inspect

T = TypeVar("T")


class Registry:
    """Simple registry for registering and retrieving classes/functions."""
    
    def __init__(self, name: str):
        """Initialize registry.
        
        Args:
            name: Registry name
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class or function.
        
        Args:
            name: Optional name to register under (defaults to class/function name)
            
        Returns:
            Decorator function
        """
        def decorator(obj: Type[T]) -> Type[T]:
            key = name if name is not None else obj.__name__
            self._registry[key] = obj
            return obj
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """Get a registered class or function.
        
        Args:
            name: Registered name
            
        Returns:
            Registered class or function
            
        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            raise KeyError(f"{name} not found in registry {self.name}. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def create(self, name: str, **kwargs) -> T:
        """Create an instance of a registered class.
        
        Args:
            name: Registered name
            **kwargs: Arguments to pass to constructor
            
        Returns:
            Instance of registered class
        """
        cls = self.get(name)
        sig = inspect.signature(cls.__init__)
        # Filter kwargs to only those accepted by the constructor
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return cls(**filtered_kwargs)
    
    def list(self) -> list[str]:
        """List all registered names.
        
        Returns:
            List of registered names
        """
        return list(self._registry.keys())

