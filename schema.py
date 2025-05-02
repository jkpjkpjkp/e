import sqlite3
import argparse

def print_schema(db_path):
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query sqlite_master for table schemas
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas = cursor.fetchall()

        # Print each schema
        if schemas:
            print("Database Schema:")
            for schema in schemas:
                print(schema[0])
        else:
            print("No tables found in the database.")

        # Close the connection
        conn.close()
    except sqlite3.Error as e:
        print(f"Error accessing database: {e}")


def main():
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check the number of graphs in the graph table
    cursor.execute("SELECT COUNT(*) FROM graph")
    num_graphs = cursor.fetchone()[0]
    print(f"Number of graphs: {num_graphs}")
    
    if num_graphs >= 2:
        # Check if there are graphs without any runs
        cursor.execute("""
            SELECT COUNT(*) 
            FROM graph g 
            LEFT JOIN run r ON g.id = r.graph_id 
            WHERE r.graph_id IS NULL
        """)
        num_graphs_without_runs = cursor.fetchone()[0]
        if num_graphs_without_runs > 0:
            print(f"Warning: {num_graphs_without_runs} graphs have no runs.")
        
        # Check the number of graphs with at least one nonzero score run
        cursor.execute("SELECT COUNT(DISTINCT graph_id) FROM run WHERE score != 0")
        num_graphs_with_nonzero = cursor.fetchone()[0]
        print(f"Number of graphs with at least one nonzero score run: {num_graphs_with_nonzero}")
        
        if num_graphs_with_nonzero == 0:
            print("No graphs have nonzero score runs.")
            conn.close()
            return
        
        # Find the highest score across all runs
        cursor.execute("SELECT MAX(score) FROM run")
        max_score = cursor.fetchone()[0]
        
        # Find a graph that has a run with the maximum score
        cursor.execute("SELECT graph_id FROM run WHERE score = ? LIMIT 1", (max_score,))
        graph_id = cursor.fetchone()[0]
        
        # Calculate the run count and average score for this graph
        cursor.execute("SELECT COUNT(*), AVG(score) FROM run WHERE graph_id = ?", (graph_id,))
        run_count, avg_score = cursor.fetchone()
        
        # Report the results
        print(f"\nHighest scoring graph ID: {graph_id}")
        print(f"Run count: {run_count}")
        print(f"Average score: {avg_score}")
    
    else:
        # When there are fewer than 2 graphs, report as per query
        # Report total number of runs
        cursor.execute("SELECT COUNT(*) FROM run")
        total_runs = cursor.fetchone()[0]
        print(f"Total number of runs: {total_runs}")
        
        if total_runs > 0:
            # Calculate win rate (percentage of runs with nonzero scores)
            cursor.execute("SELECT COUNT(*) FROM run WHERE score != 0")
            num_wins = cursor.fetchone()[0]
            win_rate = (num_wins / total_runs) * 100
            print(f"Run win rate: {win_rate:.2f}%")
        else:
            print("No runs found.")
        
        # Find an example of a 0-scoring run
        cursor.execute("SELECT graph_id, task_id, score FROM run WHERE score = 0 LIMIT 1")
        example_run = cursor.fetchone()
        if example_run:
            print("Example of a 0-scoring run:")
            print(f"Graph ID: {example_run[0]}, Task ID: {example_run[1]}, Score: {example_run[2]}")
        else:
            print("No 0-scoring runs found.")
    
    # Close the database connection
    conn.close()


if __name__ == "__main__":
    global db_path
    parser = argparse.ArgumentParser(description="Database Schema Printer")
    parser.add_argument("--db_path", help="Path to the SQLite database file")
    args = parser.parse_args()
    db_path = args.db_path
    main()