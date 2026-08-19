private_details = [
    {
        "id": 1,
        "username": "alice",
        "email": "alice@example.com",
        "role": "admin",
        "verified": True,
        "secret_token": "tok_alc_987xyz",
        "internal_code": "USR-ADM-001"
    },
    {
        "id": 2,
        "username": "bob",
        "email": "bob@example.com",
        "role": "user",
        "verified": False,
        "secret_token": "tok_bob_123abc",
        "internal_code": "USR-STD-002"
    },
    {
        "id": 3,
        "username": "carol",
        "email": "carol@example.com",
        "role": "moderator",
        "verified": True,
        "secret_token": "tok_car_456def",
        "internal_code": "USR-MOD-003"
    },
    {
        "id": 4,
        "username": "dave",
        "email": "dave@example.com",
        "role": "user",
        "verified": True,
        "secret_token": "tok_dav_789ghi",
        "internal_code": "USR-STD-004"
    },
    {
        "id": 5,
        "username": "eve",
        "email": "eve@example.com",
        "role": "premium",
        "verified": True,
        "secret_token": "tok_eve_321jkl",
        "internal_code": "USR-PRM-005"
    }
]

public_details = [
    {
        "id": 1,
        "username": "alice",
        "email": "alice@example.com",
        "full_name": "Alice Johnson",
        "role": "admin",
        "verified": True
    },
    {
        "id": 2,
        "username": "bob",
        "email": "bob@example.com",
        "full_name": "Bob Smith",
        "role": "user",
        "verified": False
    },
    {
        "id": 3,
        "username": "carol",
        "email": "carol@example.com",
        "full_name": "Carol White",
        "role": "moderator",
        "verified": True
    },
    {
        "id": 4,
        "username": "dave",
        "email": "dave@example.com",
        "full_name": "Dave Brown",
        "role": "user",
        "verified": True
    },
    {
        "id": 5,
        "username": "eve",
        "email": "eve@example.com",
        "full_name": "Eve Black",
        "role": "premium",
        "verified": True
    }
]

orders = [
    {
        "order_id": 1001,
        "customer": "Alice Johnson",
        "items": ["Laptop", "Mouse"],
        "total_amount": 1200.50,
        "status": "processing"
    },
    {
        "order_id": 1002,
        "customer": "Bob Smith",
        "items": ["Phone", "Earbuds"],
        "total_amount": 850.00,
        "status": "shipped"
    },
    {
        "order_id": 1003,
        "customer": "Carol White",
        "items": ["Desk Chair"],
        "total_amount": 150.75,
        "status": "delivered"
    },
    {
        "order_id": 1004,
        "customer": "Dave Brown",
        "items": ["Monitor", "Keyboard"],
        "total_amount": 400.00,
        "status": "processing"
    },
    {
        "order_id": 1005,
        "customer": "Eve Black",
        "items": ["Tablet"],
        "total_amount": 300.00,
        "status": "shipped"
    },
    {
        "order_id": 1006,
        "customer": "Frank Green",
        "items": ["Printer", "Ink Cartridge"],
        "total_amount": 220.00,
        "status": "delivered"
    }
]

shipments = [
    {
        "shipment_id": "SHP-001",
        "order_id": 1001,
        "carrier": "DHL",
        "tracking_number": "DHL123456",
        "estimated_delivery": "2026-08-22",
        "status": "in transit"
    },
    {
        "shipment_id": "SHP-002",
        "order_id": 1002,
        "carrier": "FedEx",
        "tracking_number": "FDX987654",
        "estimated_delivery": "2026-08-20",
        "status": "out for delivery"
    },
    {
        "shipment_id": "SHP-003",
        "order_id": 1003,
        "carrier": "UPS",
        "tracking_number": "UPS456789",
        "estimated_delivery": "2026-08-18",
        "status": "delivered"
    },
    {
        "shipment_id": "SHP-004",
        "order_id": 1004,
        "carrier": "BlueDart",
        "tracking_number": "BD123987",
        "estimated_delivery": "2026-08-23",
        "status": "pending pickup"
    },
    {
        "shipment_id": "SHP-005",
        "order_id": 1005,
        "carrier": "DHL",
        "tracking_number": "DHL654321",
        "estimated_delivery": "2026-08-21",
        "status": "in transit"
    },
    {
        "shipment_id": "SHP-006",
        "order_id": 1006,
        "carrier": "FedEx",
        "tracking_number": "FDX321654",
        "estimated_delivery": "2026-08-17",
        "status": "delivered"
    }
]
inventory = [
    {
        "item_id": "INV-001",
        "name": "Laptop",
        "category": "Electronics",
        "stock": 25,
        "price": 1200.00,
        "supplier_secret": "SUP-ALPHA-KEY",
        "internal_notes": "Priority restock needed"
    },
    {
        "item_id": "INV-002",
        "name": "Office Chair",
        "category": "Furniture",
        "stock": 40,
        "price": 150.00,
        "supplier_secret": "SUP-BETA-KEY",
        "internal_notes": "Bulk order discount applied"
    },
    {
        "item_id": "INV-003",
        "name": "Smartphone",
        "category": "Electronics",
        "stock": 60,
        "price": 850.00,
        "supplier_secret": "SUP-GAMMA-KEY",
        "internal_notes": "High demand item"
    },
    {
        "item_id": "INV-004",
        "name": "Printer",
        "category": "Office Supplies",
        "stock": 15,
        "price": 220.00,
        "supplier_secret": "SUP-DELTA-KEY",
        "internal_notes": "Limited warranty batch"
    },
    {
        "item_id": "INV-005",
        "name": "Tablet",
        "category": "Electronics",
        "stock": 30,
        "price": 300.00,
        "supplier_secret": "SUP-EPSILON-KEY",
        "internal_notes": "Next shipment delayed"
    }
]

announcements = [
    {
        "id": "ANN-001",
        "title": "Holiday Sale",
        "message": "Enjoy up to 30% off on electronics this week!",
        "date": "2026-08-19"
    },
    {
        "id": "ANN-002",
        "title": "New Arrivals",
        "message": "Fresh stock of office chairs and desks now available.",
        "date": "2026-08-18"
    },
    {
        "id": "ANN-003",
        "title": "System Maintenance",
        "message": "Scheduled downtime on August 21 from 2 AM to 4 AM.",
        "date": "2026-08-17"
    },
    {
        "id": "ANN-004",
        "title": "Premium Membership",
        "message": "Sign up for premium to unlock exclusive discounts.",
        "date": "2026-08-16"
    },
    {
        "id": "ANN-005",
        "title": "Free Shipping",
        "message": "Free shipping on all orders above $500 until August 25.",
        "date": "2026-08-15"
    }
]
