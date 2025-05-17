from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[List[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: Dict[str, object] = field(default_factory=dict)
    prev_agent: Optional[object] = None

    def summarize(self) -> str:
        import yaml
        
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        return yaml.dump(data) 