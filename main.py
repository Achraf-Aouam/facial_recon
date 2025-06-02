
import base64
import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, Form, Depends, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select, func 
from pydantic import BaseModel
from typing import List , Optional 

from database import create_db_and_tables, get_session
from models import Users as User, FaceEncoding

app = FastAPI(title="Facial Recognition API")


app.mount("/static", StaticFiles(directory="templates"), name="static")


RECOGNITION_THRESHOLD = 0.6  


class ImageRequest(BaseModel):
    image_data: str 

class OnboardResponse(BaseModel):
    message: str
    user_id: int
    face_encoding_id: int

class RecognitionResponse(BaseModel):
    status: str 
    recognized_as: str 
    distance: Optional[float] = None 
    message: Optional[str] = None 


@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

def process_image_and_get_embedding(image_data_base64: str) -> Optional[np.ndarray]:
    """Helper function to decode image and get face embedding."""
    try:
        if "," in image_data_base64:
            header, encoded = image_data_base64.split(",", 1)
        else:
            encoded = image_data_base64
            
        decoded_image = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)

    if not face_locations:
        return None 
    if len(face_locations) > 1:
        
        
        
        pass 

    face_encodings = face_recognition.face_encodings(rgb_img, [face_locations[0]]) 
    if not face_encodings: 
        return None
        
    return face_encodings[0]


@app.post("/onboard", response_model=OnboardResponse)
def onboard_user(request: ImageRequest, session: Session = Depends(get_session)):
    name = request.name 
    
    
    
    
    
    
    if not hasattr(request, 'name') or not request.name:
         raise HTTPException(status_code=400, detail="Name is required for onboarding.")

    
    try:
        if "," in request.image_data:
            header, encoded = request.image_data.split(",", 1)
        else:
            encoded = request.image_data
        decoded_image = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in the image for onboarding.")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with a single face for onboarding.")

    embedding = face_recognition.face_encodings(rgb_img, face_locations)[0]
    

    statement = select(User).where(User.name == request.name)
    db_user = session.exec(statement).first()

    if not db_user:
        db_user = User(name=request.name)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
    else:
        
        
        
        pass


    face_encoding_entry = FaceEncoding(embedding=embedding.tolist(), user_id=db_user.id)
    session.add(face_encoding_entry)
    session.commit()
    session.refresh(face_encoding_entry)

    return OnboardResponse(
        message=f"Successfully onboarded {request.name}.",
        user_id=db_user.id,
        face_encoding_id=face_encoding_entry.id
    )


class OnboardUserRequest(BaseModel):
    name: str
    image_data: str


@app.post("/onboard_user", response_model=OnboardResponse) 
def onboard_user_endpoint(request: OnboardUserRequest, session: Session = Depends(get_session)):
    
    try:
        if "," in request.image_data:
            header, encoded = request.image_data.split(",", 1)
        else:
            encoded = request.image_data
        decoded_image = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in the image for onboarding.")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide an image with a single face for onboarding.")

    embedding = face_recognition.face_encodings(rgb_img, face_locations)[0]
    

    statement = select(User).where(User.name == request.name)
    db_user = session.exec(statement).first()

    if not db_user:
        db_user = User(name=request.name)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
    
    face_encoding_entry = FaceEncoding(embedding=embedding.tolist(), user_id=db_user.id)
    session.add(face_encoding_entry)
    session.commit()
    session.refresh(face_encoding_entry)

    return OnboardResponse(
        message=f"Successfully onboarded {request.name}.",
        user_id=db_user.id,
        face_encoding_id=face_encoding_entry.id
    )


@app.post("/recognize", response_model=RecognitionResponse)
def recognize_face(request: ImageRequest, session: Session = Depends(get_session)):
    query_embedding_np = process_image_and_get_embedding(request.image_data)

    if query_embedding_np is None:
        return RecognitionResponse(status="error", recognized_as="Unknown", message="No face detected in the provided image.")

    query_embedding_list: List[float] = query_embedding_np.tolist()

    
    
    
    
    
    
    count_statement = select(func.count(FaceEncoding.id))
    total_encodings = session.exec(count_statement).one_or_none() 

    if not total_encodings or total_encodings == 0:
         return RecognitionResponse(
            status="error",
            recognized_as="Unknown",
            message="No face encodings found in the database to compare against."
        )

    
    
    
    
    
    
    stmt = (
        select(User, FaceEncoding, FaceEncoding.embedding.l2_distance(query_embedding_list).label("distance"))
        .join(User, FaceEncoding.user_id == User.id) 
        .order_by(FaceEncoding.embedding.l2_distance(query_embedding_list)) 
        .limit(1)
    )
    
    result = session.exec(stmt).first()

    if result:
        matched_user, matched_encoding, distance = result
        if distance < RECOGNITION_THRESHOLD:
            return RecognitionResponse(
                status="success",
                recognized_as=matched_user.name,
                distance=float(distance) 
            )
        else:
            return RecognitionResponse(
                status="no_match",
                recognized_as="Unknown",
                distance=float(distance),
                message=f"Closest match found: {matched_user.name}, but distance {distance:.4f} is above threshold {RECOGNITION_THRESHOLD}."
            )
    else:
        
        
        return RecognitionResponse(
            status="error",
            recognized_as="Unknown",
            message="No matching faces found in the database (query returned no results)."
        )