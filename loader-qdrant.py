import dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader

dotenv.load_dotenv()

data_path = './data/reviews.csv'
loader = CSVLoader(file_path=data_path)
data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
url = "https://3fa96c12-d0f4-4ced-9d86-eade203acf26.us-east4-0.gcp.cloud.qdrant.io"
api_key = "BsJbt2itmp59QSnemH5vHuhloje697QrE7IozDNUwzBftYO2ZVl3sA"

qdrant = QdrantVectorStore.from_documents(
    data,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="hospital_data",
)