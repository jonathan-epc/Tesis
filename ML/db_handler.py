import os  # Import os to check if file exists
import sqlite3


def manage_optuna_trials(
    database_path="studies/study24iab.db",
    action="delete_last",  # 'delete_last', 'delete_all', 'delete_from'
    start_trial_id=None,  # Required for 'delete_from' action
    confirm=True,  # Set to False to skip confirmation prompts
):
    """
    Manages trials in an Optuna SQLite database.

    Allows deleting the last trial, all trials, or trials from a specific ID onwards.

    Args:
        database_path (str): Path to the SQLite database file.
        action (str): The operation to perform:
                      - 'delete_last': Deletes the single trial with the highest trial_id.
                      - 'delete_all': Deletes all trials and associated data.
                      - 'delete_from': Deletes trials with trial_id >= start_trial_id.
        start_trial_id (int, optional): The starting trial_id for the 'delete_from'
                                        action. Defaults to None.
        confirm (bool): If True, prompt the user for confirmation before deleting.
                        Defaults to True.

    Returns:
        bool: True if the operation was successful or canceled by the user,
              False if an error occurred.
    """
    if not os.path.exists(database_path):
        print(f"Error: Database file not found at '{database_path}'")
        return False

    # Validate action and arguments
    valid_actions = ["delete_last", "delete_all", "delete_from"]
    if action not in valid_actions:
        print(f"Error: Invalid action '{action}'. Valid actions are: {valid_actions}")
        return False

    if action == "delete_from" and (
        not isinstance(start_trial_id, int) or start_trial_id < 0
    ):
        print(
            "Error: 'delete_from' action requires a valid non-negative integer for 'start_trial_id'."
        )
        return False

    print("--- Optuna Trial Management ---")
    print(f"Database: {database_path}")
    print(f"Action: {action}")
    if action == "delete_from":
        print(f"Start Trial ID: {start_trial_id}")
    print("-----------------------------")

    try:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = (
                sqlite3.Row
            )  # Enables fetching rows as dictionaries.
            cursor = connection.cursor()

            trial_ids_to_delete = []
            confirmation_message = ""

            # 1. Determine which trials to delete based on action
            if action == "delete_last":
                cursor.execute("SELECT MAX(trial_id) AS last_id FROM trials;")
                result = cursor.fetchone()
                last_trial_id = (
                    result["last_id"]
                    if result and result["last_id"] is not None
                    else None
                )

                if last_trial_id is None:
                    print("No trials found in the database.")
                    return True  # Nothing to delete, consider it success

                trial_ids_to_delete.append(last_trial_id)
                confirmation_message = f"Delete the last trial (ID: {last_trial_id})?"

                # --- Optional: Print details for the last trial ---
                cursor.execute(
                    "SELECT * FROM trials WHERE trial_id = ?", (last_trial_id,)
                )
                trial_details = cursor.fetchone()
                print(
                    "Details of the last trial:",
                    dict(trial_details) if trial_details else "Not found",
                )

                def print_associated_data(table, tid):
                    cursor.execute(f"SELECT * FROM {table} WHERE trial_id = ?", (tid,))
                    rows = cursor.fetchall()
                    data = [dict(row) for row in rows]
                    print(f"  Associated data from {table}: {data}")

                if trial_details:
                    print_associated_data("trial_params", last_trial_id)
                    print_associated_data("trial_values", last_trial_id)
                    print_associated_data("trial_intermediate_values", last_trial_id)
                # --- End Optional Print ---

            elif action == "delete_all":
                cursor.execute("SELECT trial_id FROM trials ORDER BY trial_id;")
                results = cursor.fetchall()
                if not results:
                    print("No trials found in the database.")
                    return True  # Nothing to delete

                trial_ids_to_delete = [row["trial_id"] for row in results]
                confirmation_message = (
                    f"Delete ALL {len(trial_ids_to_delete)} trials from the database?"
                )
                print(f"Found {len(trial_ids_to_delete)} trials to delete.")

            elif action == "delete_from":
                cursor.execute(
                    "SELECT trial_id FROM trials WHERE trial_id >= ? ORDER BY trial_id;",
                    (start_trial_id,),
                )
                results = cursor.fetchall()
                if not results:
                    print(f"No trials found with trial_id >= {start_trial_id}.")
                    return True  # Nothing to delete

                trial_ids_to_delete = [row["trial_id"] for row in results]
                confirmation_message = (
                    f"Delete {len(trial_ids_to_delete)} trials starting from ID {start_trial_id} "
                    f"(IDs: {trial_ids_to_delete[0]}...{trial_ids_to_delete[-1]})?"
                )
                print(
                    f"Found {len(trial_ids_to_delete)} trials to delete (IDs >= {start_trial_id})."
                )

            # 2. Ask for confirmation if required
            if not trial_ids_to_delete:
                # Should have been caught earlier, but as a safeguard
                print("No trials identified for deletion.")
                return True

            if confirm:
                user_input = (
                    input(f"CONFIRMATION: {confirmation_message} (yes/no): ")
                    .strip()
                    .lower()
                )
                if user_input != "yes":
                    print("Deletion canceled by the user.")
                    return True  # Canceled is not an error state
            else:
                print("Skipping confirmation prompt as confirm=False.")

            # 3. Perform Deletion
            print(f"Proceeding with deletion of {len(trial_ids_to_delete)} trial(s)...")

            # Define tables with trial_id foreign key
            associated_tables = [
                "trial_params",
                "trial_values",
                "trial_intermediate_values",
                "trial_user_attributes",
                "trial_system_attributes",
            ]  # Added common attributes tables

            # Create placeholders for SQL query (e.g., ?,?,?)
            placeholders = ", ".join("?" for _ in trial_ids_to_delete)
            delete_query_template = "DELETE FROM {table} WHERE trial_id IN ({phs})"

            # Delete associated data first
            for table in associated_tables:
                # Check if table exists before trying to delete (optional but safer)
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                    (table,),
                )
                if cursor.fetchone():
                    delete_query = delete_query_template.format(
                        table=table, phs=placeholders
                    )
                    cursor.execute(delete_query, trial_ids_to_delete)
                    print(f"  Deleted {cursor.rowcount} rows from {table}.")
                else:
                    print(f"  Table {table} not found, skipping.")

            # Delete from the main trials table
            delete_query = delete_query_template.format(
                table="trials", phs=placeholders
            )
            cursor.execute(delete_query, trial_ids_to_delete)
            print(f"  Deleted {cursor.rowcount} rows from trials.")

            # Commit the changes
            connection.commit()
            print("Successfully deleted specified trials and associated data.")
            return True

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        # Attempt to rollback if possible (though 'with' handles commit/rollback on exit)
        try:
            connection.rollback()
            print("Transaction rolled back.")
        except Exception:  # Ignore errors during rollback
            pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


# --- Example Usage ---
if __name__ == "__main__":
    DB_PATH = "studies/study28ddb.db"  # Replace with your actual database path

    # Delete Last Trial (with confirmation)
    manage_optuna_trials(database_path=DB_PATH, action="delete_last", confirm=True)

    # Delete Trials From ID 108 Onwards (with confirmation)
    # manage_optuna_trials(database_path=DB_PATH, action="delete_from", start_trial_id=107, confirm=True)

    # Delete All Trials (skipping confirmation)
    # manage_optuna_trials(database_path=DB_PATH, action="delete_all", confirm=False) # Use with extreme caution!

    # Example 4: Invalid Action
    # manage_optuna_trials(database_path=DB_PATH, action="delete_specific", start_trial_id=50)

    # Example 5: Missing start_trial_id for 'delete_from'
    # manage_optuna_trials(database_path=DB_PATH, action="delete_from")
