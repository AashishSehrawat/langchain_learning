from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7, max_completion_tokens=100)

# schema
json_schema = {
    'title': 'Review',
    'type': 'object',
    'properties': {
        'key_themes': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': "Write dowm all key themes discissed in the review in a list"
        },
        'summary': {
            'type': 'string',
            'description': 'A breif summary of review'
        },
        'sentiment': {
            'type': 'string',
            'enum': ['pos', 'neg'],
            'description': 'Return the sentiment statement either positive or negative'
        },
        'pros': {
            'type': ['array', 'null'],
            'items': {
                'type': 'string'
            },
            'description': 'write down all the pros inside a list'
        },
        'cons': {
            'type': ['array', 'null'],
            'items': {
                'type': 'string'
            },
            'description': 'write down all the cons inside a list'
        },
        'name': {
            'type': 'string',
            'description': 'write the name of reviwer'
        }
    }
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""The Samsung Galaxy S23 is a compact flagship that excels in performance, display quality, and camera capabilities. It's ideal for users who want a premium phone in a smaller size. 
Pros
Excellent build quality and compact design
Powerful Snapdragon 8 Gen 2 processor
Bright and vibrant 120Hz AMOLED display
Impressive camera performance, especially in daylight
Long battery life despite smaller size
Clean One UI experience with regular updates

Cons
Expensive for its size and feature set
No expandable storage
Charging speed is slower compared to competitors
Telephoto performance could be better in low light
Reviewed by
Ankit Sharma – Tech Reviewer at Smartphone Arena""")

print(result)


