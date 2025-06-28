import json

def load_tickets(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

if __name__ == "__main__":
    tickets = load_tickets("data/sample_tickets.json")
    for ticket in tickets:
        print(ticket["ticket_id"], ":", ticket["text"])
