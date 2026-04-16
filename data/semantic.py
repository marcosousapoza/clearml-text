"""Short semantic descriptions for train-partition OCEL type vocabularies.

The dictionaries below are intentionally keyed by raw event/object/qualifier
values only. They do not rename type-specific attribute tables such as
``event_attr_<type>`` or ``object_attr_<type>``.
"""

TRAINING_PARTITION_RENAME_DICTIONARY: dict[str, dict[str, object]] = {
    "bpi2017": {
        "object_types": {
            "Application": (
                "Customer-facing loan request moving through intake, review, and resolution."
            ),
            "Case_R": (
                "End-to-end credit case grouping one loan application and its related decisions."
            ),
            "Offer": (
                "Concrete loan proposal sent to the applicant and tracked through acceptance, refusal, or cancellation."
            ),
            "Workflow": (
                "Internal work item for manual or semi-automated processing steps in the credit case."
            ),
        },
        "event_types": {
            "A_Accepted": (
                "Bank approval of the loan application."
            ),
            "A_Cancelled": (
                "Cancellation of the loan application before completion."
            ),
            "A_Complete": (
                "Completion of the application record so full processing can continue."
            ),
            "A_Concept": (
                "Initial draft or concept stage of the application before formal submission."
            ),
            "A_Create Application": (
                "Creation of a new loan application."
            ),
            "A_Denied": (
                "Rejection of the loan application after assessment."
            ),
            "A_Incomplete": (
                "Application flagged as incomplete because required information or documents are missing."
            ),
            "A_Pending": (
                "Application waiting in a pending state for the next action or missing information."
            ),
            "A_Submitted": (
                "Formal submission of the loan request for processing."
            ),
            "A_Validating": (
                "Validation of the application details and supporting information."
            ),
            "O_Accepted": (
                "Applicant acceptance of a loan offer."
            ),
            "O_Cancelled": (
                "Cancellation of a loan offer that is no longer active."
            ),
            "O_Create Offer": (
                "Preparation of a new loan offer for the application."
            ),
            "O_Created": (
                "Creation of a concrete loan proposal in the system."
            ),
            "O_Refused": (
                "Applicant refusal of a loan offer."
            ),
            "O_Returned": (
                "Loan offer returned for rework or follow-up."
            ),
            "O_Sent (mail and online)": (
                "Loan offer sent through both postal and online channels."
            ),
            "O_Sent (online only)": (
                "Loan offer sent through the online channel only."
            ),
            "W_Assess potential fraud": (
                "Internal workflow step for checking possible fraud indicators in the case."
            ),
            "W_Call after offers": (
                "Follow-up calling step after a loan offer has been issued."
            ),
            "W_Call incomplete files": (
                "Contacting the applicant to resolve missing files or information."
            ),
            "W_Complete application": (
                "Operational work needed to complete the application dossier."
            ),
            "W_Handle leads": (
                "Lead-handling work around the start of application processing."
            ),
            "W_Personal Loan collection": (
                "Downstream collection work for personal loan cases."
            ),
            "W_Shortened completion": (
                "Shortened completion path used to finish selected cases more quickly."
            ),
            "W_Validate application": (
                "Internal validation work for the application and its supporting data."
            ),
        },
    },
    "bpi2019": {
        "object_types": {
            "PO": (
                "Purchase order header grouping the commercial transaction at document level."
            ),
            "POItem": (
                "Purchase order line item acting as the main unit of operational progress in the log."
            ),
            "Resource": (
                "Employee or system actor performing procurement actions."
            ),
            "Vendor": (
                "Supplier responsible for fulfilling the ordered goods or services."
            ),
        },
        "event_types": {
            "Block Purchase Order Item": (
                "Blocking a purchase order item so it cannot proceed normally."
            ),
            "Cancel Goods Receipt": (
                "Cancellation of a previously recorded goods receipt."
            ),
            "Cancel Invoice Receipt": (
                "Reversal of a previously recorded invoice receipt."
            ),
            "Cancel Subsequent Invoice": (
                "Cancellation of a subsequent invoice entry tied to the item."
            ),
            "Change Approval for Purchase Order": (
                "Change to approval status or approval data for the purchase order."
            ),
            "Change Currency": (
                "Change to the currency on the procurement document."
            ),
            "Change Delivery Indicator": (
                "Update to the delivery-related status or indicator on the item."
            ),
            "Change Final Invoice Indicator": (
                "Update to the flag indicating whether the final invoice is expected or posted."
            ),
            "Change Price": (
                "Update to the agreed price on the item."
            ),
            "Change Quantity": (
                "Change to the ordered quantity on the item."
            ),
            "Change Rejection Indicator": (
                "Update to the rejection status on the item."
            ),
            "Change Storage Location": (
                "Change to the storage location associated with the item."
            ),
            "Change payment term": (
                "Update to the payment terms for the procurement document."
            ),
            "Clear Invoice": (
                "Financial clearing of the invoice linked to the item."
            ),
            "Create Purchase Order Item": (
                "Creation of a new purchase order item."
            ),
            "Create Purchase Requisition Item": (
                "Creation of a purchase requisition item upstream of the order."
            ),
            "Delete Purchase Order Item": (
                "Deletion of a purchase order item from active processing."
            ),
            "Reactivate Purchase Order Item": (
                "Reactivation of a previously inactive or blocked purchase order item."
            ),
            "Receive Order Confirmation": (
                "Receipt of the vendor's confirmation of the order."
            ),
            "Record Goods Receipt": (
                "Recording receipt of goods or services for the item."
            ),
            "Record Invoice Receipt": (
                "Recording an incoming invoice for the item."
            ),
            "Record Service Entry Sheet": (
                "Recording a service entry sheet confirming delivered services."
            ),
            "Record Subsequent Invoice": (
                "Recording a follow-up invoice after the initial invoice."
            ),
            "Release Purchase Order": (
                "Release of the purchase order for execution after approval."
            ),
            "Remove Payment Block": (
                "Removal of a payment block so invoice settlement can continue."
            ),
            "SRM: Awaiting Approval": (
                "SRM state where the procurement document is waiting for approval."
            ),
            "SRM: Change was Transmitted": (
                "SRM transmission of a document change to the downstream execution system."
            ),
            "SRM: Complete": (
                "SRM state where the document is complete."
            ),
            "SRM: Created": (
                "SRM creation of the procurement document."
            ),
            "SRM: Deleted": (
                "SRM deletion or withdrawal of the document."
            ),
            "SRM: Document Completed": (
                "SRM milestone marking document processing as completed."
            ),
            "SRM: Held": (
                "SRM hold state for the document."
            ),
            "SRM: In Transfer to Execution Syst.": (
                "SRM transfer of the document to the execution system."
            ),
            "SRM: Incomplete": (
                "SRM state where the document remains incomplete and needs additional work."
            ),
            "SRM: Ordered": (
                "SRM state where the document has been ordered."
            ),
            "SRM: Transaction Completed": (
                "SRM completion of the procurement transaction."
            ),
            "Set Payment Block": (
                "Placement of a payment block on the invoice or item."
            ),
            "Update Order Confirmation": (
                "Update to the vendor's order confirmation information."
            ),
            "Vendor creates debit memo": (
                "Vendor-issued debit memo related to the item."
            ),
            "Vendor creates invoice": (
                "Vendor-issued invoice for the item."
            ),
        },
    },
    "order_management": {
        "object_types": {
            "customers": (
                "Buyer placing and receiving orders."
            ),
            "employees": (
                "Staff member involved in sales, packing, or shipping work."
            ),
            "items": (
                "Individual order line or picked unit handled in fulfillment."
            ),
            "orders": (
                "Customer order moving from placement through payment and delivery."
            ),
            "packages": (
                "Physical shipment prepared to deliver ordered items."
            ),
            "products": (
                "Catalog product being sold and replenished."
            ),
        },
        "event_types": {
            "confirm order": (
                "Seller confirmation of the order so fulfillment can proceed."
            ),
            "create package": (
                "Creation of a package for shipping one or more ordered items."
            ),
            "failed delivery": (
                "Failed delivery attempt where the package does not reach the customer."
            ),
            "item out of stock": (
                "Stockout event where the required product is unavailable during fulfillment."
            ),
            "package delivered": (
                "Successful delivery of the shipment to the customer."
            ),
            "pay order": (
                "Completion of payment for the order."
            ),
            "payment reminder": (
                "Reminder sent because payment for the order remains outstanding."
            ),
            "pick item": (
                "Picking an item for shipment by a warehouse or fulfillment worker."
            ),
            "place order": (
                "Placement of a new customer order."
            ),
            "reorder item": (
                "Replenishment action to restock a needed product."
            ),
            "send package": (
                "Handover of the prepared package for shipment."
            ),
        },
        "qualifiers": {
            "e2o": {
                "creates": (
                    "Object created by the event, such as a newly formed package."
                ),
                "customer": (
                    "Customer participating in the order interaction."
                ),
                "employee": (
                    "Employee carrying out the work."
                ),
                "forwarder": (
                    "Forwarding or delivery actor tied to the shipping event."
                ),
                "item": (
                    "Specific order item handled by the event."
                ),
                "order": (
                    "Order that the event belongs to."
                ),
                "packer": (
                    "Employee responsible for packing the shipment."
                ),
                "product": (
                    "Product referenced by the event, especially in stock-related actions."
                ),
                "sales person": (
                    "Salesperson associated with the order event."
                ),
                "shipped package": (
                    "Package being shipped or delivered."
                ),
                "shipper": (
                    "Shipping employee or actor responsible for dispatch."
                ),
            },
            "o2o": {
                "comprises": (
                    "Order composed of particular items."
                ),
                "contains": (
                    "Package containing specific items."
                ),
                "forwarded by": (
                    "Package forwarded by a particular employee or actor."
                ),
                "is a": (
                    "Item instantiating an underlying product."
                ),
                "packed by": (
                    "Package packed by a specific employee."
                ),
                "places": (
                    "Customer placing an order."
                ),
                "primarySalesRep": (
                    "Primary salesperson responsible for the customer or order."
                ),
                "secondarySalesRep": (
                    "Secondary supporting salesperson for the customer or order."
                ),
                "shipped by": (
                    "Package shipped by the responsible transport actor."
                ),
            },
        },
    },
    "container_logistics": {
        "object_types": {
            "Container": (
                "Shipping container being staged, loaded, and departed."
            ),
            "Customer Order": (
                "Customer transport order initiating the logistics process."
            ),
            "Forklift": (
                "Handling equipment used to move or weigh containers."
            ),
            "Handling Unit": (
                "Goods or cargo unit collected and packed into containers."
            ),
            "Transport Document": (
                "Transport paperwork coordinating containers, vehicles, and customer orders."
            ),
            "Truck": (
                "Road vehicle moving containers on site."
            ),
            "Vehicle": (
                "Booked outbound transport resource assigned to departures."
            ),
        },
        "event_types": {
            "Book Vehicles": (
                "Booking outbound transport vehicles for an upcoming shipment."
            ),
            "Bring to Loading Bay": (
                "Bringing a container to the loading bay for outbound preparation."
            ),
            "Collect Goods": (
                "Collecting goods as handling units to prepare the shipment."
            ),
            "Create Transport Document": (
                "Creation of a transport document to coordinate the shipment."
            ),
            "Depart": (
                "Shipment departure with the assigned transport resource."
            ),
            "Drive to Terminal": (
                "Truck movement carrying a container toward the terminal."
            ),
            "Load Truck": (
                "Loading a container onto a truck for internal movement."
            ),
            "Load to Vehicle": (
                "Loading a container onto the booked outbound vehicle."
            ),
            "Order Empty Containers": (
                "Ordering empty containers to satisfy customer demand."
            ),
            "Pick Up Empty Container": (
                "Picking up an empty container and bringing it into the process."
            ),
            "Place in Stock": (
                "Placing a container into stock or temporary storage."
            ),
            "Register Customer Order": (
                "Registration of a new customer transport order."
            ),
            "Reschedule Container": (
                "Rescheduling a container's transport plan or timing."
            ),
            "Weigh": (
                "Weighing a container as part of shipment preparation."
            ),
        },
        "qualifiers": {
            "e2o": {
                "CR brought to bay": (
                    "Container brought to the loading bay."
                ),
                "CR departed": (
                    "Container that departs."
                ),
                "CR laded": (
                    "Container loaded in a lading step."
                ),
                "CR loaded": (
                    "Container loaded onto another transport resource."
                ),
                "CR moved": (
                    "Container moved within the yard or terminal."
                ),
                "CR picked": (
                    "Empty container picked up."
                ),
                "CR rescheduled": (
                    "Container whose plan is being rescheduled."
                ),
                "CR stored": (
                    "Container placed into stock or storage."
                ),
                "CR weighted": (
                    "Container being weighed."
                ),
                "CRs ordered": (
                    "Containers requested when empty containers are ordered."
                ),
                "FL moved": (
                    "Forklift performing a movement action."
                ),
                "FL weighing": (
                    "Forklift involved in weighing."
                ),
                "HU collected": (
                    "Handling units collected for shipment."
                ),
                "HU loaded": (
                    "Handling units loaded into a container."
                ),
                "TD created for CO": (
                    "Newly created transport document for the customer order it serves."
                ),
                "TD with CR departure": (
                    "Transport document governing the departing container move."
                ),
                "TD with CR rescheduled": (
                    "Transport document affected by container rescheduling."
                ),
                "TR laded": (
                    "Truck receiving a loaded container."
                ),
                "TR moved": (
                    "Truck moving the container to the terminal."
                ),
                "VH departed": (
                    "Outbound vehicle used for departure."
                ),
                "VH laded": (
                    "Vehicle receiving the loaded container."
                ),
                "VHs booked for TD": (
                    "Set of vehicles booked for a transport document."
                ),
                "booked VH": (
                    "Single vehicle booked by the booking event."
                ),
                "booked VHs": (
                    "Vehicles booked together for the shipment."
                ),
                "created TD": (
                    "Transport document produced by the event."
                ),
                "ordered for TD": (
                    "Ordered containers intended for a transport document."
                ),
                "registered CO": (
                    "Customer order registered by the event."
                ),
            },
            "o2o": {
                "CR contains HU": (
                    "Container containing packed handling units."
                ),
                "CR for TD": (
                    "Container governed by a transport document."
                ),
                "Ersatz VH for TD": (
                    "Backup or substitute vehicle assigned to a transport document."
                ),
                "High-Prio VH for TD": (
                    "High-priority vehicle assigned to a transport document."
                ),
                "Regular VH for TD": (
                    "Regular assigned vehicle for a transport document."
                ),
                "TD for CO": (
                    "Transport document fulfilling a customer order."
                ),
                "TR loads CR": (
                    "Truck loading or carrying a container."
                ),
            },
        },
    },
}
