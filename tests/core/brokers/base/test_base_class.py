from datetime import datetime, timedelta, timezone
from json import JSONDecodeError, dumps
import random
from ssl import SSLError
from pandas import DataFrame, DateOffset, Series, Timedelta, Timestamp
import pytest
from unittest.mock import MagicMock, patch, Mock
from requests.models import Response
from requests.exceptions import (
    Timeout,
    ConnectionError as RequestsConnectionError,
    HTTPError,
    RequestException,
    TooManyRedirects,
)
from pandas.errors import OutOfBoundsDatetime

from core.brokers import Broker
from core.brokers import RequestTimeout, BrokerError, NetworkError
from core.brokers.base.constants import Root
from core.brokers.base.errors import InputError, ResponseError


@pytest.fixture
def mock_session():
    """
    Fixture to mock the _session attribute of the Broker class.

    This fixture patches the Broker class's _session attribute to use a mock object,
    allowing for the simulation of various behaviors and exceptions in the tests.
    """
    with patch.object(Broker, "_session", new_callable=Mock) as mock:
        yield mock


def test_fetch_success(mock_session):
    """
    Test the successful execution of the fetch method.

    This test sets up a mock response with a 200 status code and a JSON body. It verifies
    that the fetch method correctly returns this response and that the response's status
    code and text match the expected values.
    """
    mock_response = Mock(spec=Response)
    mock_response.status_code = 200
    mock_response.text = '{"key": "value"}'
    mock_session.request.return_value = mock_response

    broker = Broker()
    response = broker.fetch(method="GET", url="https://example.com")
    assert response.status_code == 200
    assert response.text == '{"key": "value"}'


def test_fetch_timeout(mock_session):
    """
    Test the fetch method's behavior when a Timeout exception is raised.

    This test simulates a timeout error by setting the mock session to raise a Timeout
    exception. It verifies that the fetch method raises a RequestTimeout exception in
    response to this error.
    """
    mock_session.request.side_effect = Timeout

    broker = Broker()
    with pytest.raises(RequestTimeout):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_connection_error(mock_session):
    """
    Test the fetch method's behavior when a ConnectionError is raised.

    This test simulates a generic connection error by setting the mock session to raise
    a RequestsConnectionError. It verifies that the fetch method raises a NetworkError
    in response to this error.
    """
    mock_session.request.side_effect = RequestsConnectionError

    broker = Broker()
    with pytest.raises(NetworkError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_connection_read_timed_out_error(mock_session):
    """
    Test the fetch method's behavior when a ConnectionError with 'Read timed out' is raised.

    This test simulates a connection read timeout error by setting the mock session to
    raise a RequestsConnectionError with the message "Read timed out". It verifies that
    the fetch method raises a RequestTimeout exception in response to this specific error.
    """
    mock_session.request.side_effect = RequestsConnectionError("Read timed out")

    broker = Broker()
    with pytest.raises(RequestTimeout):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_connection_reset_error(mock_session):
    """
    Test the fetch method's behavior when a ConnectionResetError is raised.

    This test simulates a connection reset error by setting the mock session to raise a
    ConnectionResetError. It verifies that the fetch method raises a NetworkError in
    response to this error.
    """
    mock_session.request.side_effect = ConnectionResetError

    broker = Broker()
    with pytest.raises(NetworkError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_too_many_redirects(mock_session):
    """
    Test the fetch method's behavior when a TooManyRedirects exception is raised.

    This test simulates an error due to too many redirects by setting the mock session
    to raise a TooManyRedirects exception. It verifies that the fetch method raises a
    BrokerError in response to this error.
    """
    mock_session.request.side_effect = TooManyRedirects

    broker = Broker()
    with pytest.raises(BrokerError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_ssl_error(mock_session):
    """
    Test the fetch method's behavior when an SSLError is raised.

    This test simulates an SSL error by setting the mock session to raise an SSLError.
    It verifies that the fetch method raises a BrokerError in response to this error.
    """
    mock_session.request.side_effect = SSLError

    broker = Broker()
    with pytest.raises(BrokerError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_http_error(mock_session):
    """
    Test the fetch method's behavior when an HTTPError is raised.

    This test simulates an HTTP error with a 500 status code by setting the mock session
    to raise an HTTPError. It verifies that the fetch method raises a BrokerError in
    response to this error.
    """
    mock_session.request.side_effect = HTTPError("500 Internal Server Error")

    broker = Broker()
    with pytest.raises(BrokerError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_request_exception(mock_session):
    """
    Test the fetch method's behavior when a generic RequestException is raised.

    This test simulates a generic request exception by setting the mock session to raise
    a RequestException with a random error message from a predefined list. It verifies
    that the fetch method raises a NetworkError in response to this error.
    """
    error = random.choice(["ECONNRESET", "Connection aborted.", "Connection broken:"])
    mock_session.request.side_effect = RequestException(error)

    broker = Broker()
    with pytest.raises(NetworkError):
        broker.fetch(method="GET", url="https://example.com")


def test_fetch_request_other_exception(mock_session):
    """
    Test the fetch method's behavior when an unknown RequestException is raised.

    This test simulates an unknown request exception by setting the mock session to raise
    a RequestException with a generic error message. It verifies that the fetch method
    raises a BrokerError in response to this error.
    """
    mock_session.request.side_effect = RequestException("Unknown error")

    broker = Broker()
    with pytest.raises(BrokerError):
        broker.fetch(method="GET", url="https://example.com")


@pytest.fixture
def mock_response():
    """
    Fixture that provides a mock Response object.

    This fixture creates a mock Response object that can be used in tests to simulate
    different kinds of responses from a server.
    """
    return Response()


def test_json_parser_valid_json(mock_response):
    """
    Test that the _json_parser method correctly parses a valid JSON object.

    This test verifies that the method correctly parses a simple JSON object from a
    Response object and returns the expected dictionary.
    """
    mock_response.status_code = 200
    mock_response._content = b'{"key": "value"}'
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    result = Broker._json_parser(mock_response)
    assert result == {"key": "value"}


def test_json_parser_valid_json_list(mock_response):
    """
    Test that the _json_parser method correctly parses a valid JSON array.

    This test verifies that the method correctly parses a JSON array from a Response object
    and returns the expected list of dictionaries.
    """
    mock_response.status_code = 200
    mock_response._content = b'[{"key1": "value1"}, {"key2": "value2"}]'
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    result = Broker._json_parser(mock_response)
    assert result == [{"key1": "value1"}, {"key2": "value2"}]


def test_json_parser_invalid_json(mock_response):
    """
    Test that the _json_parser method raises a ResponseError for invalid JSON.

    This test verifies that the method raises a ResponseError when it encounters a
    Response object containing invalid JSON data.
    """
    mock_response.status_code = 200
    mock_response._content = b'{"key": "value"'  # Missing closing brace
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    with pytest.raises(ResponseError) as exc_info:
        Broker._json_parser(mock_response)

    assert "Invalid JSON in response" in str(exc_info.value)


def test_json_parser_empty_response(mock_response):
    """
    Test that the _json_parser method raises a ResponseError for an empty response.

    This test verifies that the method raises a ResponseError when it encounters a
    Response object with no content (empty body).
    """
    mock_response.status_code = 200
    mock_response._content = b""
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    with pytest.raises(ResponseError) as exc_info:
        Broker._json_parser(mock_response)

    assert "Invalid JSON in response" in str(exc_info.value)


def test_json_parser_whitespace_only(mock_response):
    """
    Test that the _json_parser method raises a ResponseError for a response with only whitespace.

    This test verifies that the method raises a ResponseError when it encounters a
    Response object containing only whitespace characters in the body.
    """
    mock_response.status_code = 200
    mock_response._content = b"   \n\t  "
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    with pytest.raises(ResponseError) as exc_info:
        Broker._json_parser(mock_response)

    assert "Invalid JSON in response" in str(exc_info.value)


def test_json_parser_non_json_content(mock_response):
    """
    Test that the _json_parser method raises a ResponseError for non-JSON content.

    This test verifies that the method raises a ResponseError when it encounters a
    Response object containing non-JSON content (e.g., plain text).
    """
    mock_response.status_code = 200
    mock_response._content = b"This is not JSON"
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    with pytest.raises(ResponseError) as exc_info:
        Broker._json_parser(mock_response)

    assert "Invalid JSON in response" in str(exc_info.value)

    mock_response.status_code = 200
    mock_response._content = b"This is not JSON"
    mock_response.url = "http://test.com"
    mock_response.reason = "OK"

    with pytest.raises(ResponseError) as exc_info:
        Broker._json_parser(mock_response)

    assert "Invalid JSON in response" in str(exc_info.value)


def test_json_dumps_simple_dict():
    """Test json_dumps with a simple dictionary."""
    data = {"key": "value"}
    result = Broker.json_dumps(data)
    assert result == '{"key": "value"}'


def test_json_dumps_nested_dict():
    """Test json_dumps with a nested dictionary."""
    data = {"outer": {"inner": "value"}}
    result = Broker.json_dumps(data)
    assert result == '{"outer": {"inner": "value"}}'


def test_json_dumps_with_list():
    """Test json_dumps with a dictionary containing a list."""
    data = {"key": [1, 2, 3]}
    result = Broker.json_dumps(data)
    assert result == '{"key": [1, 2, 3]}'


def test_json_dumps_with_numbers():
    """Test json_dumps with various number types."""
    data = {"integer": 42, "float": 3.14, "negative": -10}
    result = Broker.json_dumps(data)
    assert result == '{"integer": 42, "float": 3.14, "negative": -10}'


def test_json_dumps_with_boolean():
    """Test json_dumps with boolean values."""
    data = {"true": True, "false": False}
    result = Broker.json_dumps(data)
    assert result == '{"true": true, "false": false}'


def test_json_dumps_with_null():
    """Test json_dumps with None (null in JSON)."""
    data = {"null_value": None}
    result = Broker.json_dumps(data)
    assert result == '{"null_value": null}'


def test_json_dumps_empty_dict():
    """Test json_dumps with an empty dictionary."""
    data = {}
    result = Broker.json_dumps(data)
    assert result == "{}"


def test_json_dumps_complex_structure():
    """Test json_dumps with a complex nested structure."""
    data = {
        "string": "value",
        "number": 42,
        "float": 3.14,
        "boolean": True,
        "null": None,
        "list": [1, "two", 3.0],
        "nested": {"a": 1, "b": [2, 3, 4], "c": {"d": "nested"}},
    }
    result = Broker.json_dumps(data)
    expected = dumps(data)  # Using the actual json.dumps for comparison
    assert result == expected


def test_json_dumps_non_dict_input():
    """Test json_dumps with non-dictionary input to ensure it raises a TypeError."""
    with pytest.raises(TypeError):
        Broker.json_dumps([1, 2, 3])  # List instead of dictionary


def test_on_json_response_valid_json(mock_response):
    """
    Test case for valid JSON content in the response.

    Verifies that a properly formatted JSON response is correctly parsed into a dictionary.
    """
    mock_response._content = b'{"key": "value"}'
    result = Broker.on_json_response(mock_response)
    assert result == {"key": "value"}


def test_on_json_response_invalid_json(mock_response):
    """
    Test case for invalid JSON content in the response.

    Verifies that an invalid JSON response raises a JSONDecodeError.
    """
    mock_response._content = b'{"key": "value"'  # Invalid JSON
    with pytest.raises(JSONDecodeError):
        Broker.on_json_response(mock_response)


def test_on_json_response_empty_response(mock_response):
    """
    Test case for an empty response.

    Verifies that an empty response raises a JSONDecodeError.
    """
    mock_response._content = b""  # Empty response
    with pytest.raises(JSONDecodeError):
        Broker.on_json_response(mock_response)


def test_on_json_response_non_json_content(mock_response):
    """
    Test case for non-JSON content in the response.

    Verifies that a response with non-JSON content raises a JSONDecodeError.
    """
    mock_response._content = b"This is not JSON"  # Non-JSON content
    with pytest.raises(JSONDecodeError):
        Broker.on_json_response(mock_response)


def test_generate_verified_totp_success():
    """
    Test case for successfully generating a valid TOTP.

    Verifies that a valid TOTP is generated with a length of 6 digits and consists only of digits.
    """
    totpbase = "JBSWY3DPEHPK3PXP"
    totp = Broker.generate_verified_totp(totpbase)
    assert len(totp) == 6
    assert totp.isdigit()


def test_generate_verified_totp_invalid_base():
    """
    Test case for providing an invalid base string.

    Ensures that a ValueError is raised when an invalid TOTP base string is provided.
    """
    with pytest.raises(ValueError):
        Broker.generate_verified_totp("invalid_base")


def test_generate_verified_totp_max_attempts_reached():
    """
    Test case for reaching the maximum number of attempts.

    Verifies that a ValueError is raised when the maximum number of attempts to generate a valid TOTP is exceeded.
    """
    totpbase = "JBSWY3DPEHPK3PXP"
    with patch("pyotp.TOTP.verify", return_value=False):
        with pytest.raises(
            ValueError, match="Unable to generate a valid TOTP after 3 attempts"
        ):
            Broker.generate_verified_totp(totpbase)


def test_generate_verified_totp_custom_max_attempts():
    """
    Test case for using a custom maximum number of attempts.

    Verifies that a ValueError is raised with the correct message when the maximum number of attempts is set to a custom value.
    """
    totpbase = "JBSWY3DPEHPK3PXP"
    with patch("pyotp.TOTP.verify", return_value=False):
        with pytest.raises(
            ValueError, match="Unable to generate a valid TOTP after 5 attempts"
        ):
            Broker.generate_verified_totp(totpbase, max_attempts=5)


def test_generate_verified_totp_second_attempt_success():
    """
    Test case for successful TOTP generation on the second attempt.

    Verifies that a valid TOTP is generated if it takes more than one attempt.
    """
    totpbase = "JBSWY3DPEHPK3PXP"
    with patch("pyotp.TOTP.verify", side_effect=[False, True]):
        totp = Broker.generate_verified_totp(totpbase)
        assert len(totp) == 6
        assert totp.isdigit()


def test_generate_verified_totp_empty_base():
    """
    Test case for providing an empty base string.

    Ensures that a ValueError is raised when an empty TOTP base string is provided.
    """
    with pytest.raises(ValueError):
        Broker.generate_verified_totp("")


def test_valid_totp_base():
    """
    Test case for a valid TOTP base string.

    Verifies that a valid TOTP is generated when a valid TOTP base string is used.
    """
    valid_totp = "ABCDEFGHIJKLMNOP"
    assert Broker.generate_verified_totp(valid_totp) is not None


def test_invalid_totp_base_non_base32():
    """
    Test case for a non-base32 TOTP base string.

    Ensures that a ValueError is raised when a TOTP base string containing non-base32 characters is provided.
    """
    invalid_totp = "ABCDEFGH12345678"
    with pytest.raises(ValueError, match="Invalid TOTP base"):
        Broker.generate_verified_totp(invalid_totp)


def test_empty_totp_base():
    """
    Test case for an empty TOTP base string.

    Ensures that a ValueError is raised when an empty TOTP base string is provided.
    """
    with pytest.raises(ValueError, match="Invalid TOTP base"):
        Broker.generate_verified_totp("")


def test_totp_base_with_padding():
    """
    Test case for a TOTP base string with padding characters.

    Ensures that a ValueError is raised when a TOTP base string contains padding characters (e.g., '=').
    """
    valid_totp_with_padding = "ABCDEFGHIJKLMNOP===="
    with pytest.raises(ValueError, match="Invalid TOTP base"):
        Broker.generate_verified_totp(valid_totp_with_padding)


def test_totp_base_lowercase():
    """
    Test case for a TOTP base string with lowercase letters.

    Verifies that a TOTP is generated when the base string contains lowercase letters.
    """
    lowercase_totp = "abcdefghijklmnop"
    assert Broker.generate_verified_totp(lowercase_totp) is not None


def test_totp_base_mixed_case():
    """
    Test case for a TOTP base string with mixed case letters.

    Verifies that a TOTP is generated when the base string contains mixed case letters.
    """
    mixed_case_totp = "AbCdEfGhIjKlMnOp"
    assert Broker.generate_verified_totp(mixed_case_totp) is not None


def test_totp_base_with_spaces():
    """
    Test case for a TOTP base string with spaces.

    Ensures that a ValueError is raised when a TOTP base string contains spaces.
    """
    totp_with_spaces = "ABCD EFGH IJKL MNOP"
    with pytest.raises(ValueError, match="Invalid TOTP base"):
        Broker.generate_verified_totp(totp_with_spaces)


def test_totp_base_with_special_characters():
    """
    Test case for a TOTP base string with special characters.

    Ensures that a ValueError is raised when a TOTP base string contains special characters.
    """
    totp_with_special_chars = "ABCDEFGH!@#$%^&*"
    with pytest.raises(ValueError, match="Invalid TOTP base"):
        Broker.generate_verified_totp(totp_with_special_chars)


@pytest.fixture
def mock_read_json():
    """
    Fixture to mock the pandas.read_json function.

    This fixture patches the read_json function in the Broker's module,
    allowing you to simulate different return values or behaviors for
    reading JSON data in tests.
    """
    with patch("core.brokers.base.base.read_json") as mock:
        yield mock


@pytest.fixture
def mock_read_csv():
    """
    Fixture to mock the pandas.read_csv function.

    This fixture patches the read_csv function in the Broker's module,
    allowing you to simulate different return values or behaviors for
    reading CSV data in tests.
    """
    with patch("core.brokers.base.base.read_csv") as mock:
        yield mock


def test_data_reader_json(mock_read_json):
    """
    Test reading JSON data using the data_reader method.

    This test verifies that when the filetype is 'json', the data_reader method
    correctly calls the read_json function and returns the expected DataFrame.
    """
    # Mock return value
    mock_df = DataFrame({"key": ["value"]})
    mock_read_json.return_value = mock_df

    result = Broker.data_reader("http://example.com/data.json", "json")

    mock_read_json.assert_called_once_with("http://example.com/data.json")
    assert isinstance(result, DataFrame)
    assert result.equals(mock_df)


def test_data_reader_csv(mock_read_csv):
    """
    Test reading CSV data using the data_reader method with default parameters.

    This test verifies that when the filetype is 'csv' and no additional parameters
    are provided, the data_reader method correctly calls the read_csv function with
    default arguments and returns the expected DataFrame.
    """
    # Mock return value
    mock_df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    mock_read_csv.return_value = mock_df

    result = Broker.data_reader("http://example.com/data.csv", "csv")

    mock_read_csv.assert_called_once_with(
        "http://example.com/data.csv",
        dtype=None,
        sep=",",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    assert isinstance(result, DataFrame)
    assert result.equals(mock_df)


def test_data_reader_csv_with_dtype(mock_read_csv):
    """
    Test reading CSV data using the data_reader method with specified data types.

    This test verifies that the data_reader method correctly passes the dtype argument
    to the read_csv function and returns the expected DataFrame.
    """
    # Mock return value
    mock_df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    mock_read_csv.return_value = mock_df

    dtype = {"col1": "int", "col2": "str"}

    result = Broker.data_reader("http://example.com/data.csv", "csv", dtype=dtype)

    mock_read_csv.assert_called_once_with(
        "http://example.com/data.csv",
        dtype=dtype,
        sep=",",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    assert isinstance(result, DataFrame)
    assert result.equals(mock_df)


def test_data_reader_csv_with_sep(mock_read_csv):
    """
    Test reading CSV data using the data_reader method with a custom separator.

    This test verifies that the data_reader method correctly uses the custom separator
    provided in the sep argument and returns the expected DataFrame.
    """
    # Mock return value
    mock_df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    mock_read_csv.return_value = mock_df

    result = Broker.data_reader("http://example.com/data.csv", "csv", sep="|")

    mock_read_csv.assert_called_once_with(
        "http://example.com/data.csv",
        dtype=None,
        sep="|",
        on_bad_lines="skip",
        encoding_errors="ignore",
    )
    assert isinstance(result, DataFrame)
    assert result.equals(mock_df)


def test_data_reader_csv_with_col_names(mock_read_csv):
    """
    Test reading CSV data using the data_reader method with custom column names.

    This test verifies that the data_reader method correctly passes the col_names argument
    to the read_csv function and returns the expected DataFrame with the specified column names.
    """
    # Mock return value
    mock_df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    mock_read_csv.return_value = mock_df

    col_names = ["col1", "col2"]

    result = Broker.data_reader(
        "http://example.com/data.csv", "csv", col_names=col_names
    )

    mock_read_csv.assert_called_once_with(
        "http://example.com/data.csv",
        dtype=None,
        sep=",",
        names=col_names,
    )
    assert isinstance(result, DataFrame)
    assert result.equals(mock_df)


def test_data_reader_invalid_filetype():
    """
    Test that the data_reader method raises an InputError for invalid file types.

    This test verifies that when an unsupported file type is provided, the data_reader method
    raises an InputError with the appropriate error message.
    """
    with pytest.raises(
        InputError, match="Wrong Filetype: xml, the possible values are: 'json', 'csv'"
    ):
        Broker.data_reader("http://example.com/data.xml", "xml")


def test_data_frame_with_valid_data():
    """
    Test data_frame method with valid list data.

    This test verifies that the data_frame method correctly converts a list of dictionaries
    into a DataFrame with the expected structure and content.
    """
    data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]
    result = Broker.data_frame(data)

    expected_df = DataFrame(data)

    assert isinstance(result, DataFrame)
    assert result.equals(expected_df)


def test_data_frame_with_empty_list():
    """
    Test data_frame method with an empty list.

    This test verifies that the data_frame method correctly handles an empty list input
    by returning an empty DataFrame.
    """
    data = []
    result = Broker.data_frame(data)

    expected_df = DataFrame(data)

    assert isinstance(result, DataFrame)
    assert result.equals(expected_df)
    assert result.empty


def test_data_frame_with_mixed_data():
    """
    Test data_frame method with mixed data types in the list.

    This test verifies that the data_frame method correctly handles a list containing
    mixed data types, converting it into a DataFrame with the appropriate structure.
    """
    data = [
        {"col1": 1, "col2": "a"},
        {"col1": 2, "col2": None},
        {"col1": None, "col2": "c"},
    ]
    result = Broker.data_frame(data)

    expected_df = DataFrame(data)

    assert isinstance(result, DataFrame)
    assert result.equals(expected_df)


def test_pd_datetime_with_string():
    """
    Test pd_datetime with a string input representing a date.

    The test verifies that a string date is correctly converted to a Timestamp object.
    """
    result = Broker.pd_datetime("2023-05-01")
    assert isinstance(result, Timestamp)
    assert result == Timestamp("2023-05-01")


def test_pd_datetime_with_integer():
    """
    Test pd_datetime with an integer input representing a Unix timestamp in seconds.

    The test verifies that an integer timestamp (in seconds) is correctly converted to a Timestamp object.
    """
    result = Broker.pd_datetime(1620000000, unit="s")
    assert isinstance(result, Timestamp)
    assert result == Timestamp("2021-05-03 00:00:00")


def test_pd_datetime_with_float():
    """
    Test pd_datetime with a float input representing a Unix timestamp in seconds.

    The test verifies that a float timestamp (in seconds, with fractional seconds) is correctly converted to a Timestamp object,
    including fractional seconds.
    """
    result = Broker.pd_datetime(1620000000.5, unit="s")
    assert isinstance(result, Timestamp)
    assert result == Timestamp("2021-05-03 00:00:00.500000")


def test_pd_datetime_with_numeric_string():
    """
    Test pd_datetime with a numeric string input representing a Unix timestamp in seconds.

    The test verifies that a numeric string timestamp is correctly converted to a Timestamp object.
    """
    result = Broker.pd_datetime("1620000000", unit="s")
    assert isinstance(result, Timestamp)
    assert result == Timestamp("2021-05-03 00:00:00")


def test_pd_datetime_with_timezone():
    """
    Test pd_datetime with a string input and a specified timezone.

    The test verifies that a date string with a specified timezone is correctly converted to a Timestamp object with timezone information.
    """
    result = Broker.pd_datetime("2023-05-01", tz="UTC")
    assert isinstance(result, Timestamp)
    assert result == Timestamp("2023-05-01", tz="UTC")


def test_pd_datetime_with_different_units():
    """
    Test pd_datetime with various units to ensure correct conversion.

    The test verifies that different units ('D', 's', 'ms', 'us', 'ns') are handled correctly by pd_datetime.
    """
    units = ["D", "s", "ms", "us", "ns"]
    for unit in units:
        result = Broker.pd_datetime(1, unit=unit)
        assert isinstance(result, Timestamp)


def test_pd_datetime_out_of_bounds():
    """
    Test pd_datetime with a value that is out of bounds for the specified unit.

    The test ensures that pd_datetime raises an OutOfBoundsDatetime exception when given a timestamp that exceeds the allowable range.
    """
    with pytest.raises(OutOfBoundsDatetime):
        Broker.pd_datetime(10**20, unit="ns")


def test_pd_datetime_invalid_input():
    """
    Test pd_datetime with an invalid date string.

    The test ensures that pd_datetime raises a ValueError when given a string that cannot be parsed into a valid datetime.
    """
    with pytest.raises(ValueError):
        Broker.pd_datetime("invalid date")


def test_pd_datetime_invalid_unit():
    """
    Test pd_datetime with an invalid unit.

    The test ensures that pd_datetime raises a ValueError when given an unrecognized unit.
    """
    with pytest.raises(ValueError):
        Broker.pd_datetime(1, unit="invalid")


def test_datetime_strp_valid_input():
    """
    Test that Broker.datetime_strp correctly parses a valid datetime string
    using the specified format and returns a datetime object.
    """
    result = Broker.datetime_strp("2023-05-01 14:30:00", "%Y-%m-%d %H:%M:%S")
    assert isinstance(result, datetime)
    assert result == datetime(2023, 5, 1, 14, 30, 0)


def test_datetime_strp_different_format():
    """
    Test that Broker.datetime_strp correctly parses a datetime string
    with a different format and returns the correct datetime object.
    """
    result = Broker.datetime_strp("01/05/2023", "%d/%m/%Y")
    assert isinstance(result, datetime)
    assert result == datetime(2023, 5, 1)


def test_datetime_strp_with_microseconds():
    """
    Test that Broker.datetime_strp correctly parses a datetime string
    that includes microseconds and returns the correct datetime object.
    """
    result = Broker.datetime_strp("2023-05-01 14:30:00.123456", "%Y-%m-%d %H:%M:%S.%f")
    assert isinstance(result, datetime)
    assert result == datetime(2023, 5, 1, 14, 30, 0, 123456)


def test_datetime_strp_invalid_date():
    """
    Test that Broker.datetime_strp raises a ValueError when given an invalid date,
    such as a month that doesn't exist.
    """
    with pytest.raises(ValueError):
        Broker.datetime_strp("2023-13-01", "%Y-%m-%d")  # Invalid month


def test_datetime_strp_mismatched_format():
    """
    Test that Broker.datetime_strp raises a ValueError when the datetime string
    does not match the format specified.
    """
    with pytest.raises(ValueError):
        Broker.datetime_strp(
            "2023-05-01", "%d/%m/%Y"
        )  # Format doesn't match the string


def test_datetime_strp_empty_string():
    """
    Test that Broker.datetime_strp raises a ValueError when given an empty string.
    """
    with pytest.raises(ValueError):
        Broker.datetime_strp("", "%Y-%m-%d")


def test_datetime_strp_none_input():
    """
    Test that Broker.datetime_strp raises a TypeError when None is passed as the datetime string.
    """
    with pytest.raises(TypeError):
        Broker.datetime_strp(None, "%Y-%m-%d")


def test_datetime_strp_invalid_format_string():
    """
    Test that Broker.datetime_strp raises a ValueError when the format string includes
    more specifiers than are present in the datetime string.
    """
    with pytest.raises(ValueError):
        Broker.datetime_strp(
            "2023-05-01", "%Y-%m-%d %H:%M:%S"
        )  # More format specifiers than in the string


def test_from_timestamp_valid_input():
    """
    Test from_timestamp with a valid integer Unix timestamp.

    The test verifies that an integer Unix timestamp is correctly converted to a naive datetime object,
    and that the resulting datetime object has the same timestamp value.
    """
    result = Broker.from_timestamp(1620000000)  # May 3, 2021 12:00:00 AM GMT
    assert isinstance(result, datetime)
    assert result.timestamp() == 1620000000


def test_from_timestamp_zero():
    """
    Test from_timestamp with a Unix timestamp of zero.

    The test verifies that a Unix timestamp of zero (representing January 1, 1970 12:00:00 AM GMT) is
    correctly converted to a naive datetime object, and that the resulting datetime object has a timestamp of zero.
    """
    result = Broker.from_timestamp(0)  # January 1, 1970 12:00:00 AM GMT
    assert isinstance(result, datetime)
    assert result.timestamp() == 0


def test_from_timestamp_current_time():
    """
    Test from_timestamp with the current time's Unix timestamp.

    The test verifies that the current Unix timestamp is correctly converted to a naive datetime object,
    and that the resulting datetime object is within one second of the current time.
    """
    current_timestamp = datetime.now().timestamp()
    result = Broker.from_timestamp(current_timestamp)
    assert isinstance(result, datetime)
    assert abs(result.timestamp() - current_timestamp) < 1  # Allow 1 second difference


def test_from_timestamp_negative():
    """
    Test from_timestamp with a negative Unix timestamp.

    This should raise a ValueError as negative timestamps are not supported.
    """
    with pytest.raises(ValueError):
        Broker.from_timestamp(-1000000)  # A date before 1970


def test_from_timestamp_float():
    """
    Test from_timestamp with a float Unix timestamp.

    The test verifies that a float Unix timestamp (including fractional seconds) is correctly converted to a naive datetime object,
    and that the resulting datetime object has a timestamp value close to the input, allowing for minor differences.
    """
    result = Broker.from_timestamp(1620000000.5)
    assert isinstance(result, datetime)
    assert (
        abs(result.timestamp() - 1620000000.5) < 0.001
    )  # Allow small float difference


def test_from_timestamp_string():
    """
    Test from_timestamp with a string input.

    The test verifies that a TypeError is raised when the input is a string, as the function should only accept numerical timestamps.
    """
    with pytest.raises(ValueError):
        Broker.from_timestamp("1620000000")


def test_from_timestamp_none():
    """
    Test from_timestamp with a None input.

    The test verifies that a TypeError is raised when the input is None, as the function should only accept numerical timestamps.
    """
    with pytest.raises(ValueError):
        Broker.from_timestamp(None)


def test_from_timestamp_very_large_number():
    """
    Test from_timestamp with a very large Unix timestamp.

    The test verifies that a ValueError is raised when a very large Unix timestamp (potentially causing overflow) is provided.
    """
    with pytest.raises(ValueError):
        Broker.from_timestamp(
            2**63
        )  # A very large number that should cause an overflow


def test_from_timestamp_very_small_number():
    """
    Test from_timestamp with a very small (negative) Unix timestamp.

    The test verifies that a ValueError is raised when a very small Unix timestamp (potentially causing an error) is provided.
    """
    with pytest.raises(ValueError):
        Broker.from_timestamp(
            -(2**63)
        )  # A very small number that should cause an error


def test_from_timestamp_timezone():
    """
    Test from_timestamp with a Unix timestamp and verify its relationship to UTC.

    The test verifies that the resulting naive datetime object is within 24 hours of the corresponding UTC time
    and has no timezone information.
    """
    timestamp = 1620000000
    result = Broker.from_timestamp(timestamp)
    utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    assert result.tzinfo is None  # The result should be naive (no timezone info)
    assert (
        abs((result - utc_time.replace(tzinfo=None)).total_seconds()) < 24 * 3600
    )  # Should be within 24 hours of UTC time


def test_current_datetime_type():
    """
    Test current_datetime to ensure it returns a datetime object.
    """
    result = Broker.current_datetime()
    assert isinstance(result, datetime)


def test_current_datetime_accuracy():
    """
    Test current_datetime to ensure the returned datetime is close to the actual current time.

    The test verifies that the returned datetime is within 1 second of the system's current time.
    """
    current_time = datetime.now()
    result = Broker.current_datetime()
    assert isinstance(result, datetime)
    assert (
        abs((result - current_time).total_seconds()) < 1
    )  # Allow up to 1 second difference


def test_current_datetime_utc_offset():
    """
    Test current_datetime to ensure the returned datetime is in the local timezone and not UTC.

    The test verifies that the returned datetime object does not have a timezone (i.e., it is naive).
    """
    result = Broker.current_datetime()
    assert isinstance(result, datetime)
    assert result.tzinfo is None  # Ensure datetime is naive (no timezone info)


def test_time_delta_add():
    """
    Test time_delta to add days to a datetime object.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 10
    dtformat = "%Y-%m-%d"
    result = Broker.time_delta(dt, delta_days, dtformat, default="add")
    expected = (dt + timedelta(days=delta_days)).strftime(dtformat)
    assert result == expected


def test_time_delta_subtract():
    """
    Test time_delta to subtract days from a datetime object.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 10
    dtformat = "%Y-%m-%d"
    result = Broker.time_delta(dt, delta_days, dtformat, default="sub")
    expected = (dt - timedelta(days=delta_days)).strftime(dtformat)
    assert result == expected


def test_time_delta_add_days():
    """
    Test time_delta to add days to a datetime object.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 5
    dtformat = "%Y-%m-%d"
    result = Broker.time_delta(dt, delta_days, dtformat, default="add")
    expected = (dt + timedelta(days=delta_days)).strftime(dtformat)
    assert result == expected


def test_time_delta_invalid_default():
    """
    Test time_delta with an invalid 'default' value.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 5
    dtformat = "%Y-%m-%d"
    try:
        Broker.time_delta(dt, delta_days, dtformat, default="invalid")
    except InputError as e:
        assert str(e) == "Wrong default: invalid, the possible values are 'sub', 'add'"
    else:
        assert False, "InputError not raised with an invalid default value."


def test_time_delta_invalid_format():
    """
    Test time_delta with the default behavior (subtract days) and a valid datetime format.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 5
    dtformat = "%Y-%m-%d"  # Valid format
    result = Broker.time_delta(dt, delta_days, dtformat)
    expected = (dt - timedelta(days=delta_days)).strftime(dtformat)
    assert result == expected


def test_time_delta_edge_case():
    """
    Test time_delta with a zero delta.
    """
    dt = datetime(2023, 1, 1)
    delta_days = 0
    dtformat = "%Y-%m-%d"
    result = Broker.time_delta(dt, delta_days, dtformat)
    expected = dt.strftime(dtformat)
    assert result == expected


def test_dateoffset_with_days_and_hours():
    """
    Test creating a DateOffset with days and hours.

    This test verifies that the Broker.dateoffset method correctly creates
    a DateOffset object when provided with days and hours as arguments.
    """
    offset = Broker.dateoffset(days=1, hours=2)
    assert isinstance(offset, DateOffset)
    assert offset.kwds == {"days": 1, "hours": 2}


def test_dateoffset_with_months_and_years():
    """
    Test creating a DateOffset with months and years.

    This test checks if the Broker.dateoffset method properly creates
    a DateOffset object with the provided months and years arguments.
    """
    offset = Broker.dateoffset(months=6, years=1)
    assert isinstance(offset, DateOffset)
    assert offset.kwds == {"months": 6, "years": 1}


def test_dateoffset_invalid_arguments():
    """
    Test creating a DateOffset with invalid arguments.

    This test ensures that the Broker.dateoffset method raises a ValueError
    when provided with an invalid argument.
    """
    with pytest.raises(ValueError):
        Broker.dateoffset(invalid_argument=5)


def test_dateoffset_with_no_arguments():
    """
    Test creating a DateOffset with no arguments.

    This test verifies that the Broker.dateoffset method can create a
    DateOffset object even when no arguments are provided.
    """
    offset = Broker.dateoffset()
    assert isinstance(offset, DateOffset)


def test_dateoffset_with_weeks():
    """
    Test creating a DateOffset with weeks.

    This test checks if the Broker.dateoffset method can correctly create
    a DateOffset object using weeks as an argument.
    """
    offset = Broker.dateoffset(weeks=2)
    assert isinstance(offset, DateOffset)
    assert offset.kwds == {"weeks": 2}


def test_dateoffset_combination():
    """
    Test creating a DateOffset with a combination of different arguments.

    This test verifies that the Broker.dateoffset method can correctly
    handle multiple arguments (days, weeks, months, years) and produce
    a valid DateOffset object.
    """
    offset = Broker.dateoffset(days=3, weeks=1, months=2, years=1)
    assert isinstance(offset, DateOffset)
    assert offset.kwds == {"days": 3, "weeks": 1, "months": 2, "years": 1}


def test_concatenate_dataframes_valid_input():
    """
    Test concatenating valid DataFrame inputs.

    This test checks that the Broker.concatenate_dataframes method correctly
    concatenates a list of valid pandas DataFrames into a single DataFrame.
    """
    df1 = DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = DataFrame({"A": [5, 6], "B": [7, 8]})
    result = Broker.concatenate_dataframes([df1, df2]).reset_index(drop=True)
    expected = DataFrame({"A": [1, 2, 5, 6], "B": [3, 4, 7, 8]})
    assert result.equals(expected)


def test_concatenate_dataframes_empty_list():
    """
    Test concatenating an empty list of DataFrames.

    This test ensures that the Broker.concatenate_dataframes method raises
    a ValueError when an empty list is provided as input.
    """
    with pytest.raises(ValueError, match="Input list is empty"):
        Broker.concatenate_dataframes([])


def test_concatenate_dataframes_invalid_element():
    """
    Test concatenating a list with an invalid element.

    This test checks that the Broker.concatenate_dataframes method raises
    a TypeError when an element in the list is not a pandas DataFrame.
    """
    df1 = DataFrame({"A": [1, 2], "B": [3, 4]})
    invalid_element = {"A": [5, 6], "B": [7, 8]}
    with pytest.raises(TypeError, match="All elements must be pandas DataFrames"):
        Broker.concatenate_dataframes([df1, invalid_element])


def test_concatenate_dataframes_with_kwargs():
    """
    Test concatenating DataFrames with additional kwargs.

    This test checks if the Broker.concatenate_dataframes method correctly
    handles additional keyword arguments, such as ignore_index, passed to
    the concat function.
    """
    df1 = DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = DataFrame({"A": [5, 6], "B": [7, 8]})
    result = Broker.concatenate_dataframes([df1, df2], ignore_index=True)
    expected = DataFrame({"A": [1, 2, 5, 6], "B": [3, 4, 7, 8]}).reset_index(drop=True)
    assert result.equals(expected)


def test_filter_future_dates_with_list():
    """
    Test the `filter_future_dates` method with a list of date strings.

    This test ensures that the `filter_future_dates` method correctly filters a list of date strings, keeping only the future dates and removing the past dates.
    """
    # Test with a list of date strings
    # Create a Series with dates that are dynamically relative to the current date
    current_date = datetime.now().date()
    data = Series(
        [
            (current_date - Timedelta(days=5)).strftime("%Y-%m-%d"),
            (current_date + Timedelta(days=3)).strftime("%Y-%m-%d"),
            (current_date - Timedelta(days=10)).strftime("%Y-%m-%d"),
            (current_date + Timedelta(days=15)).strftime("%Y-%m-%d"),
        ]
    )
    # Expected result should include only future dates
    expected = [
        (current_date + Timedelta(days=3)).strftime("%Y-%m-%d"),
        (current_date + Timedelta(days=15)).strftime("%Y-%m-%d"),
    ]
    result = Broker.filter_future_dates(data)
    assert result == expected


def test_filter_future_dates_with_series():
    """
    Test the `filter_future_dates` method with a dynamic pandas Series of date strings.

    This Series contains a mix of future and past dates relative to the current date.
    """
    # Create a Series with dates that are dynamically relative to the current date
    current_date = datetime.now().date()
    data = Series(
        [
            (current_date - Timedelta(days=5)).strftime("%Y-%m-%d"),
            (current_date + Timedelta(days=3)).strftime("%Y-%m-%d"),
            (current_date - Timedelta(days=10)).strftime("%Y-%m-%d"),
            (current_date + Timedelta(days=15)).strftime("%Y-%m-%d"),
        ]
    )

    # Expected result should include only future dates
    expected = [
        (current_date + Timedelta(days=3)).strftime("%Y-%m-%d"),
        (current_date + Timedelta(days=15)).strftime("%Y-%m-%d"),
    ]

    # Call the method
    result = Broker.filter_future_dates(data)

    # Assert the result matches the expected future dates
    assert result == expected, f"Expected {expected}, but got {result}"


def test_filter_future_dates_with_empty_list():
    """
    Test the `filter_future_dates` method with an empty list.

    This test ensures that the `filter_future_dates` method correctly handles an empty input list, and returns an empty list.
    """
    # Test with an empty list
    data = []
    expected = []
    result = Broker.filter_future_dates(data)
    assert result == expected


def test_filter_future_dates_with_empty_series():
    """
    Test the `filter_future_dates` method with an empty pandas Series.

    This test ensures that the `filter_future_dates` method correctly handles an empty input Series, and returns an empty list.
    """
    # Test with an empty pandas Series
    data = Series([])
    expected = []
    result = Broker.filter_future_dates(data)
    assert result == expected


def test_filter_future_dates_no_future_dates():
    """
    Test the `filter_future_dates` method with a list of dates all in the past.

    This test ensures that the `filter_future_dates` method correctly returns an empty list when provided with a list of dates that are all in the past.
    """
    # Test with dates all in the past
    data = ["2024-07-01", "2024-06-30"]
    expected = []
    result = Broker.filter_future_dates(data)
    assert result == expected


def test_filter_future_dates_invalid_dates():
    """
    Test the `filter_future_dates` method with invalid date formats.

    This test ensures that the `filter_future_dates` method correctly raises a `ValueError` when provided with a list of dates that contain invalid formats.
    """
    # Test with invalid date formats
    data = ["not-a-date", "2024-08-01"]
    with pytest.raises(ValueError):
        Broker.filter_future_dates(data)


def test_filter_future_dates_edge_case():
    """
    Test the `filter_future_dates` method with an edge case where the current date is included in the input list.

    This test ensures that the `filter_future_dates` method correctly handles the case where the current date is present in the input list, and that the returned list is sorted.
    """
    fixed_now = "2024-08-11"
    data = [fixed_now, "2024-08-15"]
    expected = sorted([fixed_now, "2024-08-15"])

    with patch("pandas.Timestamp.now", return_value=Timestamp(fixed_now)):
        result = Broker.filter_future_dates(data)

    assert result == expected


@patch("core.brokers.base.base.req_session")
def test_successful_request(mock_session):
    """
    Test the successful execution of the download_expiry_dates_nfo method.

    This test ensures that when the request is successful, the method correctly
    processes the response and stores the expiry dates.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "records": {"expiryDates": ["2024-08-29", "2024-09-26", "2024-10-31"]}
    }
    mock_session.return_value.request.return_value = mock_response

    Broker.expiry_dates = {}
    Broker.nfo_url = "https://example.com/nfo"
    Broker.cookies = {"cookie": "value"}

    Broker.download_expiry_dates_nfo("NIFTY")

    assert "NIFTY" in Broker.expiry_dates
    assert len(Broker.expiry_dates["NIFTY"]) == 3


@patch("core.brokers.base.base.req_session")
@patch("core.brokers.base.base.popen")
def test_fallback_to_curl(mock_popen, mock_session):
    """
    Test the fallback to curl when the initial request fails.

    This test verifies that if the initial request fails, the method
    falls back to using curl and still processes the response correctly.
    """
    mock_session.return_value.request.side_effect = Exception("Request failed")

    mock_popen.return_value.read.return_value = dumps(
        {"records": {"expiryDates": ["2024-08-29", "2024-09-26"]}}
    )

    Broker.expiry_dates = {}
    Broker.nfo_url = "https://example.com/nfo"
    Broker.cookies = {"cookie": "value"}

    Broker.download_expiry_dates_nfo("BANKNIFTY")

    assert "BANKNIFTY" in Broker.expiry_dates
    assert len(Broker.expiry_dates["BANKNIFTY"]) == 2


@patch("core.brokers.base.base.req_session")
@patch("core.brokers.base.base.popen")
@patch("core.brokers.base.base.sleep")
def test_all_attempts_fail(mock_sleep, mock_popen, mock_session):
    """
    Test the behavior when all download attempts fail.

    This test ensures that when both the initial request and the curl fallback
    fail, the method handles the failure gracefully and does not add any
    expiry dates for the symbol.
    """
    mock_session.return_value.request.side_effect = Exception("Request failed")
    mock_popen.return_value.read.side_effect = Exception("Curl failed")

    Broker.expiry_dates = {}
    Broker.nfo_url = "https://example.com/nfo"
    Broker.cookies = {"cookie": "value"}

    Broker.download_expiry_dates_nfo("FAIL")

    assert "FAIL" not in Broker.expiry_dates
    assert mock_sleep.call_count == 5


@patch("core.brokers.base.base.req_session")
def test_filter_future_dates(mock_session):
    """
    Test the filtering of future dates in the download_expiry_dates_nfo method.

    This test verifies that the method correctly filters out past dates and
    only keeps future dates (including today) in the expiry_dates dictionary.
    """
    today = datetime.now().date()
    past_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    future_date1 = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    future_date2 = (today + timedelta(days=30)).strftime("%Y-%m-%d")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "records": {
            "expiryDates": [
                past_date,
                today.strftime("%Y-%m-%d"),
                future_date1,
                future_date2,
            ]
        }
    }
    mock_session.return_value.request.return_value = mock_response

    Broker.expiry_dates = {}
    Broker.nfo_url = "https://example.com/nfo"
    Broker.cookies = {"cookie": "value"}

    Broker.download_expiry_dates_nfo("TEST")

    assert "TEST" in Broker.expiry_dates
    assert len(Broker.expiry_dates["TEST"]) == 2
    assert past_date not in Broker.expiry_dates["TEST"]
    assert today.strftime("%Y-%m-%d") not in Broker.expiry_dates["TEST"]
    assert future_date1 in Broker.expiry_dates["TEST"]
    assert future_date2 in Broker.expiry_dates["TEST"]


@pytest.fixture
def mock_response_data():
    today = datetime.now().date()
    return {
        "Table1": [
            {"ExpiryDate": (today + timedelta(days=1)).strftime("%Y-%m-%d")},
            {"ExpiryDate": (today + timedelta(days=8)).strftime("%Y-%m-%d")},
            {"ExpiryDate": (today + timedelta(days=15)).strftime("%Y-%m-%d")},
        ]
    }


@patch("core.brokers.base.base.req_session")
def test_download_expiry_dates_bfo_success(mock_session, mock_response_data):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_session.return_value.request.return_value = mock_response

    Broker.bfo_url = "https://example.com/bfo"
    Broker.expiry_dates = {}

    Broker.download_expiry_dates_bfo(Root.SENSEX)

    assert Root.SENSEX in Broker.expiry_dates
    assert len(Broker.expiry_dates[Root.SENSEX]) == 3
    mock_session.return_value.request.assert_called_once()


@patch("core.brokers.base.base.req_session")
def test_download_expiry_dates_bfo_different_roots(mock_session, mock_response_data):
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_session.return_value.request.return_value = mock_response

    Broker.bfo_url = "https://example.com/bfo"
    Broker.expiry_dates = {}

    Broker.download_expiry_dates_bfo(Root.SENSEX)
    Broker.download_expiry_dates_bfo(Root.BANKEX)

    assert Root.SENSEX in Broker.expiry_dates
    assert Root.BANKEX in Broker.expiry_dates
    assert mock_session.return_value.request.call_count == 2


@patch("core.brokers.base.base.req_session")
@patch("core.brokers.base.base.popen")
def test_download_expiry_dates_bfo_fallback_to_curl(
    mock_popen, mock_session, mock_response_data
):
    mock_session.return_value.request.side_effect = Exception("Request failed")

    mock_popen.return_value.read.return_value = dumps(mock_response_data)

    Broker.bfo_url = "https://example.com/bfo"
    Broker.expiry_dates = {}

    Broker.download_expiry_dates_bfo(Root.SENSEX)

    assert Root.SENSEX in Broker.expiry_dates
    assert len(Broker.expiry_dates[Root.SENSEX]) == 3
    mock_session.return_value.request.assert_called_once()
    mock_popen.assert_called_once()


@patch("core.brokers.base.base.req_session")
@patch("core.brokers.base.base.popen")
@patch("core.brokers.base.base.sleep")
def test_download_expiry_dates_bfo_all_attempts_fail(
    mock_sleep, mock_popen, mock_session
):
    mock_session.return_value.request.side_effect = Exception("Request failed")
    mock_popen.return_value.read.side_effect = Exception("Curl failed")

    Broker.bfo_url = "https://example.com/bfo"
    Broker.expiry_dates = {}

    Broker.download_expiry_dates_bfo(Root.SENSEX)

    assert Root.SENSEX not in Broker.expiry_dates
    assert mock_sleep.call_count == 5


@patch("core.brokers.base.base.req_session")
def test_download_expiry_dates_bfo_filter_past_dates(mock_session):
    today = datetime.now().date()
    mock_response_data = {
        "Table1": [
            {"ExpiryDate": (today - timedelta(days=1)).strftime("%Y-%m-%d")},
            {"ExpiryDate": today.strftime("%Y-%m-%d")},
            {"ExpiryDate": (today + timedelta(days=1)).strftime("%Y-%m-%d")},
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_session.return_value.request.return_value = mock_response

    Broker.bfo_url = "https://example.com/bfo"
    Broker.expiry_dates = {}

    Broker.download_expiry_dates_bfo(Root.SENSEX)

    assert Root.SENSEX in Broker.expiry_dates
    assert len(Broker.expiry_dates[Root.SENSEX]) == 1  # Not Today and future date only
    assert all(
        date >= today.strftime("%Y-%m-%d") for date in Broker.expiry_dates[Root.SENSEX]
    )


# @pytest.fixture
# def sample_dataframe():
#     today = datetime.now().date()
#     expiry1 = today + timedelta(days=1)
#     expiry2 = today + timedelta(days=8)
#     expiry3 = today + timedelta(days=15)

#     data = {
#         'Token': [1, 2, 3, 4],
#         'Symbol': ['BANKNIFTY23316CE', 'NIFTY23316PE', 'FINNIFTY23316CE', 'MIDCPNIFTY23316PE'],
#         'Expiry': [expiry1, expiry2, expiry3, expiry1],
#         'Option': ['CE', 'PE', 'CE', 'PE'],
#         'StrikePrice': [34500, 17000, 18000, 7500],
#         'LotSize': [25, 50, 40, 75],
#         'Root': ['BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
#         'TickSize': [0.05, 0.05, 0.05, 0.05]
#     }
#     return DataFrame(data)

# @patch.object(Broker, 'download_expiry_dates_nfo')
# @patch.object(Broker, 'download_expiry_dates_bfo')
# def test_jsonify_expiry_basic_structure(mock_bfo, mock_nfo, sample_dataframe):
#     Broker.expiry_dates = {
#         Root.BNF: [str(datetime.now().date() + timedelta(days=i)) for i in range(1, 4)],
#         Root.NF: [str(datetime.now().date() + timedelta(days=i)) for i in range(1, 4)],
#         Root.FNF: [str(datetime.now().date() + timedelta(days=i)) for i in range(1, 4)],
#         Root.MIDCPNF: [str(datetime.now().date() + timedelta(days=i)) for i in range(1, 4)]
#     }

#     result = Broker.jsonify_expiry(sample_dataframe)

#     assert set(result.keys()) == {WeeklyExpiry.CURRENT, WeeklyExpiry.NEXT, WeeklyExpiry.FAR, WeeklyExpiry.EXPIRY, WeeklyExpiry.LOTSIZE}
#     assert set(result[WeeklyExpiry.CURRENT].keys()) == {Root.BNF, Root.NF, Root.FNF, Root.MIDCPNF, Root.SENSEX, Root.BANKEX}
#     assert set(result[WeeklyExpiry.EXPIRY].keys()) == {Root.BNF, Root.NF, Root.FNF, Root.MIDCPNF, Root.SENSEX, Root.BANKEX}
#     assert set(result[WeeklyExpiry.LOTSIZE].keys()) == {Root.BNF, Root.NF, Root.FNF, Root.MIDCPNF, Root.SENSEX, Root.BANKEX}

# # def test_jsonify_expiry_lotsize(sample_dataframe):
# #     result = Broker.jsonify_expiry(sample_dataframe)

# #     assert result[WeeklyExpiry.LOTSIZE][Root.BNF] == 25
# #     assert result[WeeklyExpiry.LOTSIZE][Root.NF] == 50
# #     assert result[WeeklyExpiry.LOTSIZE][Root.FNF] == 40
# #     assert result[WeeklyExpiry.LOTSIZE][Root.MIDCPNF] == 75

# # def test_jsonify_expiry_expiry_dates(sample_dataframe):
# #     result = Broker.jsonify_expiry(sample_dataframe)

# #     assert len(result[WeeklyExpiry.EXPIRY][Root.BNF]) == 1
# #     assert len(result[WeeklyExpiry.EXPIRY][Root.NF]) == 1
# #     assert len(result[WeeklyExpiry.EXPIRY][Root.FNF]) == 1
# #     assert len(result[WeeklyExpiry.EXPIRY][Root.MIDCPNF]) == 1

# # def test_jsonify_expiry_option_data(sample_dataframe):
# #     result = Broker.jsonify_expiry(sample_dataframe)

# #     assert 'CE' in result[WeeklyExpiry.CURRENT][Root.BNF]
# #     assert 'PE' in result[WeeklyExpiry.CURRENT][Root.NF]
# #     assert 34500 in result[WeeklyExpiry.CURRENT][Root.BNF]['CE']
# #     assert 17000 in result[WeeklyExpiry.CURRENT][Root.NF]['PE']

# # @patch.object(Broker, 'download_expiry_dates_nfo')
# # @patch.object(Broker, 'download_expiry_dates_bfo')
# # def test_jsonify_expiry_download_calls(mock_bfo, mock_nfo, sample_dataframe):
# #     Broker.expiry_dates = {}
# #     Broker.jsonify_expiry(sample_dataframe)

# #     assert mock_nfo.call_count == 4  # BNF, NF, FNF, MIDCPNF
# #     assert mock_bfo.call_count == 2  # SENSEX, BANKEX

# # def test_jsonify_expiry_empty_dataframe():
# #     empty_df = DataFrame(columns=['Token', 'Symbol', 'Expiry', 'Option', 'StrikePrice', 'LotSize', 'Root', 'TickSize'])
# #     result = Broker.jsonify_expiry(empty_df)

# #     assert all(not result[WeeklyExpiry.CURRENT][root] for root in Root)
# #     assert all(not result[WeeklyExpiry.EXPIRY][root] for root in Root)
# #     assert all(isna(result[WeeklyExpiry.LOTSIZE][root]) for root in Root)

# # @pytest.mark.parametrize("root", [Root.BNF, Root.NF, Root.FNF, Root.MIDCPNF])
# # def test_jsonify_expiry_specific_root(root, sample_dataframe):
# #     specific_df = sample_dataframe[sample_dataframe['Root'] == root.value]
# #     result = Broker.jsonify_expiry(specific_df)

# #     assert result[WeeklyExpiry.CURRENT][root]
# #     assert result[WeeklyExpiry.EXPIRY][root]
# #     assert not isna(result[WeeklyExpiry.LOTSIZE][root])

# # def test_jsonify_expiry_sorting(sample_dataframe):
# #     unsorted_df = sample_dataframe.sort_values(by=['Expiry'], ascending=False)
# #     result = Broker.jsonify_expiry(unsorted_df)

# #     for root in [Root.BNF, Root.NF, Root.FNF, Root.MIDCPNF]:
# #         if result[WeeklyExpiry.EXPIRY][root]:
# #             assert result[WeeklyExpiry.EXPIRY][root] == sorted(result[WeeklyExpiry.EXPIRY][root])
