from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from core.security import verify_password, hash_password, create_access_token, get_current_user, TokenData

router = APIRouter(tags=["auth"])

USERS_DB = {
    "analyst":        {"password": hash_password("analyst123"),  "role": "analyst"},
    "senior_analyst": {"password": hash_password("senior123"),   "role": "senior_analyst"},
    "admin":          {"password": hash_password("admin123"),    "role": "admin"},
}

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

@router.post("/auth/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": form_data.username, "role": user["role"]})
    return TokenResponse(access_token=token, token_type="bearer", role=user["role"], username=form_data.username)

@router.get("/auth/me")
async def get_me(current_user: TokenData = Depends(get_current_user)):
    return {"username": current_user.username, "role": current_user.role}