"""
API Response Validation Layer for Perpetual Ethical Oversight MAS

Provides:
- Response schema validation
- Status code checking
- Data integrity validation
- Error categorization and handling
- Retry-aware validation with circuit breaker integration
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class APIErrorType(Enum):
    """Categorized API error types"""
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NOT_FOUND_ERROR = "not_found_error"
    UNAUTHORIZED_ERROR = "unauthorized_error"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    UNKNOWN_ERROR = "unknown_error"


class HTTPStatus(Enum):
    """HTTP status codes"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    RATE_LIMIT = 429
    SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


@dataclass
class APIErrorDetail:
    """Details about an API error"""
    error_type: APIErrorType
    http_status: Optional[int] = None
    error_message: str = ""
    error_code: Optional[str] = None
    retry_after: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_retryable(self) -> bool:
        """Check if error is retryable"""
        retryable_types = {
            APIErrorType.NETWORK_ERROR,
            APIErrorType.TIMEOUT_ERROR,
            APIErrorType.RATE_LIMIT_ERROR,
            APIErrorType.SERVER_ERROR,
            APIErrorType.SERVICE_UNAVAILABLE,
        }
        return self.error_type in retryable_types
    
    def __str__(self) -> str:
        return (
            f"{self.error_type.value} "
            f"(HTTP {self.http_status}): {self.error_message}"
        )


@dataclass
class APIResponse:
    """Validated API response"""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    raw_body: str = ""
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    error_detail: Optional[APIErrorDetail] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            'status_code': self.status_code,
            'is_valid': self.is_valid,
            'data': self.data,
            'validation_errors': self.validation_errors,
            'timestamp': self.timestamp.isoformat(),
        }


class APIValidator:
    """
    Validates API responses for correctness and integrity
    
    Features:
    - Status code validation
    - Response schema validation
    - Data type checking
    - Required field validation
    - Custom validation rules
    """
    
    def __init__(self, api_name: str):
        """
        Initialize API validator
        
        Args:
            api_name: Name of the API being validated
        """
        self.api_name = api_name
        self.validation_rules = []
        self.expected_status_codes = {HTTPStatus.OK.value}
        self.required_fields = set()
        self.field_types = {}
    
    def add_status_code(self, code: Union[int, HTTPStatus]):
        """Add acceptable status code"""
        if isinstance(code, HTTPStatus):
            code = code.value
        self.expected_status_codes.add(code)
    
    def add_required_field(self, field_name: str):
        """Add required field for response"""
        self.required_fields.add(field_name)
    
    def add_field_type(self, field_name: str, field_type: Type):
        """Add expected field type"""
        self.field_types[field_name] = field_type
    
    def add_validation_rule(self, rule_func):
        """
        Add custom validation rule
        
        Args:
            rule_func: Function(response_data) -> (is_valid: bool, error_message: str)
        """
        self.validation_rules.append(rule_func)
    
    def validate_response(
        self,
        status_code: int,
        response_body: Any,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """
        Validate API response
        
        Args:
            status_code: HTTP status code
            response_body: Response body (dict, list, or string)
            headers: Response headers
        
        Returns:
            APIResponse with validation results
        """
        response = APIResponse(
            status_code=status_code,
            data=response_body,
            headers=headers or {},
        )
        
        # Check status code
        if status_code not in self.expected_status_codes:
            response.is_valid = False
            error_detail = self._categorize_http_error(status_code)
            response.error_detail = error_detail
            response.validation_errors.append(
                f"Unexpected status code: {status_code} "
                f"(expected: {self.expected_status_codes})"
            )
            return response
        
        # Parse JSON if string
        data = response_body
        if isinstance(response_body, str):
            try:
                data = json.loads(response_body)
                response.raw_body = response_body
            except json.JSONDecodeError as e:
                response.is_valid = False
                response.validation_errors.append(
                    f"Invalid JSON: {str(e)}"
                )
                response.error_detail = APIErrorDetail(
                    error_type=APIErrorType.MALFORMED_RESPONSE,
                    http_status=status_code,
                    error_message=f"Failed to parse response: {str(e)}"
                )
                return response
        
        response.data = data
        
        # Validate structure
        if isinstance(data, dict):
            structure_errors = self._validate_dict_structure(data)
            response.validation_errors.extend(structure_errors)
            if structure_errors:
                response.is_valid = False
        
        # Run custom validation rules
        for rule in self.validation_rules:
            try:
                is_valid, error_msg = rule(data)
                if not is_valid:
                    response.validation_errors.append(error_msg)
                    response.is_valid = False
            except Exception as e:
                response.validation_errors.append(
                    f"Validation rule error: {str(e)}"
                )
                response.is_valid = False
        
        return response
    
    def _validate_dict_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate dictionary structure"""
        errors = []
        
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")
            elif field_name in self.field_types:
                expected_type = self.field_types[field_name]
                actual_value = data[field_name]
                if not isinstance(actual_value, expected_type):
                    errors.append(
                        f"Field '{field_name}' has wrong type: "
                        f"expected {expected_type.__name__}, "
                        f"got {type(actual_value).__name__}"
                    )
        
        return errors
    
    def _categorize_http_error(self, status_code: int) -> APIErrorDetail:
        """Categorize HTTP error"""
        categorization = {
            400: APIErrorType.VALIDATION_ERROR,
            401: APIErrorType.UNAUTHORIZED_ERROR,
            403: APIErrorType.UNAUTHORIZED_ERROR,
            404: APIErrorType.NOT_FOUND_ERROR,
            429: APIErrorType.RATE_LIMIT_ERROR,
            500: APIErrorType.SERVER_ERROR,
            503: APIErrorType.SERVICE_UNAVAILABLE,
        }
        
        error_type = categorization.get(status_code, APIErrorType.UNKNOWN_ERROR)
        return APIErrorDetail(
            error_type=error_type,
            http_status=status_code,
            error_message=f"HTTP {status_code} error"
        )


class ResponseSchemaValidator:
    """
    Validates responses against predefined schemas
    
    Schemas can define:
    - Required fields
    - Field types
    - Field value ranges
    - Nested object structures
    """
    
    def __init__(self):
        self.schemas = {}
    
    def register_schema(self, api_name: str, schema: Dict[str, Any]):
        """
        Register a response schema
        
        Example schema:
        {
            'fields': {
                'id': {'type': 'string', 'required': True},
                'data': {'type': 'list', 'required': True},
                'timestamp': {'type': 'string', 'required': False},
            }
        }
        """
        self.schemas[api_name] = schema
    
    def validate(self, api_name: str, response_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate response against registered schema
        
        Args:
            api_name: Name of API with registered schema
            response_data: Response data to validate
        
        Returns:
            (is_valid, error_messages)
        """
        if api_name not in self.schemas:
            return True, []  # No schema registered
        
        schema = self.schemas[api_name]
        errors = []
        
        # Validate fields
        if 'fields' in schema:
            for field_name, field_spec in schema['fields'].items():
                is_required = field_spec.get('required', False)
                expected_type = field_spec.get('type', 'any')
                
                if field_name not in response_data:
                    if is_required:
                        errors.append(f"Missing required field: {field_name}")
                else:
                    value = response_data[field_name]
                    if not self._check_type(value, expected_type):
                        errors.append(
                            f"Field '{field_name}' has wrong type: "
                            f"expected {expected_type}, got {type(value).__name__}"
                        )
        
        return len(errors) == 0, errors
    
    def _check_type(self, value: Any, type_name: str) -> bool:
        """Check if value matches type name"""
        type_map = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'any': object,
        }
        
        expected_type = type_map.get(type_name, object)
        return isinstance(value, expected_type)


# Predefined validators for common APIs
ARXIV_VALIDATOR = APIValidator("arxiv")
ARXIV_VALIDATOR.add_required_field("feed")
ARXIV_VALIDATOR.add_field_type("feed", dict)

REDDIT_VALIDATOR = APIValidator("reddit")
REDDIT_VALIDATOR.add_required_field("data")
REDDIT_VALIDATOR.add_field_type("data", dict)
REDDIT_VALIDATOR.add_required_field("data.children")
REDDIT_VALIDATOR.add_field_type("data.children", list)

LEGAL_VALIDATOR = APIValidator("legal_api")
LEGAL_VALIDATOR.add_required_field("results")
LEGAL_VALIDATOR.add_field_type("results", list)
