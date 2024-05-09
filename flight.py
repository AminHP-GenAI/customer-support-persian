from typing import List, Dict, Type, Optional

import json
import pytz
from datetime import datetime

from langchain_core.runnables import ensure_config
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from database import Database


class FlightManager:

    def __init__(self, db: Database) -> None:
        self.db = db

    def fetch_user_flight_information(self, passenger_id: str) -> List[Dict]:
        connection = self.db.get_connection()
        cursor = connection.cursor()

        query = """
        SELECT 
            t.ticket_no, t.book_ref,
            f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
            bp.seat_no, tf.fare_conditions
        FROM 
            tickets t
            JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
            JOIN flights f ON tf.flight_id = f.flight_id
            JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
        WHERE 
            t.passenger_id = ?
        """

        cursor.execute(query, (passenger_id,))
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        connection.close()

        return results

    def search_flights(
        self,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict]:
        connection = self.db.get_connection()
        cursor = connection.cursor()

        query = "SELECT * FROM flights WHERE 1 = 1"
        params = []

        if departure_airport:
            query += " AND departure_airport = ?"
            params.append(departure_airport)

        if arrival_airport:
            query += " AND arrival_airport = ?"
            params.append(arrival_airport)

        if start_time:
            query += " AND scheduled_departure >= ?"
            params.append(start_time)

        if end_time:
            query += " AND scheduled_departure <= ?"
            params.append(end_time)

        query += " LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        connection.close()

        return results

    def update_ticket_to_new_flight(
        self,
        passenger_id: str,
        ticket_no: str,
        new_flight_id: int,
    ) -> str:
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
            (new_flight_id,),
        )
        new_flight = cursor.fetchone()

        if not new_flight:
            cursor.close()
            connection.close()
            return "Invalid `new_flight_id` provided."

        column_names = [column[0] for column in cursor.description]
        new_flight_dict = dict(zip(column_names, new_flight))
        timezone = pytz.timezone("Etc/GMT-3")
        current_time = datetime.now(tz=timezone)
        departure_time = datetime.strptime(
            new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
        )
        time_until = (departure_time - current_time).total_seconds()
        if time_until < (3 * 3600):
            cursor.close()
            connection.close()
            return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        current_flight = cursor.fetchone()
        if not current_flight:
            cursor.close()
            connection.close()
            return "No existing ticket found for the given `ticket_no`."

        # Check the signed-in user actually has this ticket
        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            cursor.close()
            connection.close()
            return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

        # In a real application, you'd likely add additional checks here to enforce business logic,
        # like "does the new departure airport match the current ticket", etc.
        # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
        # it's inevitably going to get things wrong, so you **also** need to ensure your
        # API enforces valid behavior
        cursor.execute(
            "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
            (new_flight_id, ticket_no),
        )
        connection.commit()

        cursor.close()
        connection.close()
        return "Ticket successfully updated to new flight."

    def cancel_ticket(self, passenger_id: str, ticket_no: str) -> str:
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
        )
        existing_ticket = cursor.fetchone()
        if not existing_ticket:
            cursor.close()
            connection.close()
            return "No existing ticket found for the given `ticket_no`."

        # Check the signed-in user actually has this ticket
        cursor.execute(
            "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
            (ticket_no, passenger_id),
        )
        current_ticket = cursor.fetchone()
        if not current_ticket:
            cursor.close()
            connection.close()
            return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

        cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
        connection.commit()

        cursor.close()
        connection.close()
        return "Ticket successfully cancelled."

    def get_tools(self) -> Dict[str, BaseTool]:
        tools = [
            FetchUserFlightInformationTool(flight_manager=self),
            SearchFlightsTool(flight_manager=self),
            UpdateTicketToNewFlightTool(flight_manager=self),
            CancelTicketTool(flight_manager=self),
        ]
        return {tool.name: tool for tool in tools}



class FetchUserFlightInformationToolInput(BaseModel):
    pass


class FetchUserFlightInformationTool(BaseTool):

    name = 'fetch_user_flight_information_tool'
    description = (
        "Fetch all tickets for the user along with corresponding flight information and seat assignments.\n"
        "Returns A list of dictionaries where each dictionary contains the ticket details, "
        "associated flight details, and the seat assignments for each ticket belonging to the user."
    )
    args_schema: Type[BaseModel] = FetchUserFlightInformationToolInput
    return_direct: bool = False

    flight_manager: FlightManager

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        config = ensure_config()
        configuration = config.get('configurable', {})
        passenger_id = configuration.get('passenger_id', None)
        if not passenger_id:
            raise ValueError("No `passenger_id` configured.")

        results = self.flight_manager.fetch_user_flight_information(passenger_id)
        return json.dumps(results)



class SearchFlightsToolInput(BaseModel):
    departure_airport: Optional[str] = Field(description="should be emprty or the departure airport code")
    arrival_airport: Optional[str] = Field(description="should be emprty or the departure arrival code")
    start_time: Optional[str] = Field(
        description=
        # "it must be in the format `yyyy-MM-dd` or `yyyy-MM-dd HH:mm` or empty string, "
        "must be empty or in iso format, "
        "specifies date or datetime for search lower band"
    )
    end_time: Optional[str] = Field(
        description=
        # "it must be in the format `yyyy-MM-dd` or `yyyy-MM-dd HH:mm` or empty string, "
        "must be empty or in iso format, "
        "specifies date or datetime for search upper band"
    )
    limit: Optional[int] = Field(description="specifies the maximum number of search results")


class SearchFlightsTool(BaseTool):

    name = 'search_flights_tool'
    description = (
        "Search for flights based on departure airport, arrival airport, and departure time range.\n"
        "Returns A list of dictionaries where each dictionary contains the flight details."
    )
    args_schema: Type[BaseModel] = SearchFlightsToolInput
    return_direct: bool = False

    flight_manager: FlightManager

    def _run(
        self,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 20,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        departure_airport = departure_airport or None
        arrival_airport = arrival_airport or None
        start_time = datetime.fromisoformat(start_time) if start_time else None
        end_time = datetime.fromisoformat(end_time) if end_time else None

        results = self.flight_manager.search_flights(
            departure_airport, arrival_airport, start_time, end_time, limit
        )
        return json.dumps(results)



class UpdateTicketToNewFlightToolInput(BaseModel):
    ticket_no: str = Field(description="should be the user's ticket number")
    new_flight_id: int = Field(description="should be a new flight id")


class UpdateTicketToNewFlightTool(BaseTool):

    name = 'update_ticket_to_new_flight_tool'
    description = (
        "Update the user's ticket to a new valid flight."
    )
    args_schema: Type[BaseModel] = UpdateTicketToNewFlightToolInput
    return_direct: bool = False

    flight_manager: FlightManager

    def _run(
        self,
        ticket_no: str, new_flight_id: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        config = ensure_config()
        configuration = config.get('configurable', {})
        passenger_id = configuration.get('passenger_id', None)
        if not passenger_id:
            raise ValueError("No `passenger_id` configured.")

        results = self.flight_manager.update_ticket_to_new_flight(
            passenger_id, ticket_no, new_flight_id
        )
        return json.dumps(results)



class CancelTicketToolInput(BaseModel):
    ticket_no: str = Field(description="should be the user's ticket number")


class CancelTicketTool(BaseTool):

    name = 'cancel_ticket_tool'
    description = (
        "Cancel the user's ticket and remove it from the database."
    )
    args_schema: Type[BaseModel] = CancelTicketToolInput
    return_direct: bool = False

    flight_manager: FlightManager

    def _run(
        self,
        ticket_no: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        config = ensure_config()
        configuration = config.get('configurable', {})
        passenger_id = configuration.get('passenger_id', None)
        if not passenger_id:
            raise ValueError("No `passenger_id` configured.")

        results = self.flight_manager.cancel_ticket(passenger_id, ticket_no)
        return results
