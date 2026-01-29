import { useContext, useState, useRef } from "react";
import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import { MdPermMedia } from "react-icons/md";
import Intercept from "../Tools/refrech";
import axios from "axios";
import { AuthContext } from "../contexts/AuthContext/AuthContext";
import { NotificationManager } from "react-notifications";

function Share(props) {
  const navigate = useNavigate();
  const desc = useRef();
  const { user } = useContext(AuthContext);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false); // Add loading state
  const axiosJWT = axios.create();
  Intercept(axiosJWT);

  const submitHandler = async (e) => {
    e.preventDefault();
    if (!file) {
        NotificationManager.warning("Warning", "Photo is required", 3000);
        return;
    }

    setLoading(true);
    e.currentTarget.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("description", desc.current.value);

      // Send to your FastAPI backend instead of Cloudinary directly
      // The backend will handle the AI check AND the Cloudinary upload
      await axiosJWT.post("http://localhost:8000/api/create-post", formData, {
        headers: { 
            Authorization: "Bearer " + user.accessToken,
            "Content-Type": "multipart/form-data" 
        },
      });

      NotificationManager.success("Success", "Post created successfully (Human Verified)", 3000);
      setFile(null);
      desc.current.value = "";
      props.hideAddPostHandler();
      navigate(`/home`);

    } catch (err) {
      console.error(err);
      // Handle the specific error if the backend rejects it as AI
      if (err.response && err.response.status === 406) {
         NotificationManager.error("Blocked", "AI Content Detected! This platform is for human content only.", 5000);
      } else {
         NotificationManager.error("Error", "Something went wrong during upload.", 3000);
      }
      e.currentTarget.disabled = false;
    } finally {
        setLoading(false);
    }
  };

  return (
    <ShareContainer>
      <div className="shareWrapper">
        <div className="shareTop">
          <input
            placeholder={"What's in your mind ?"}
            className="shareInput"
            ref={desc}
            required
            disabled={loading}
          />
        </div>
        <hr className="shareHr" />
        <form className="shareBottom">
          <div className="shareOptions">
            <label htmlFor="file" className="shareOption">
              <MdPermMedia className="shareIcon" />
              <span className="shareOptionText">
                {file ? file.name : "Photo or Video"}
              </span>
              <input
                style={{ display: "none" }}
                type="file"
                id="file"
                accept=".png,.jpeg,.jpg"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </label>
          </div>
          <button 
            className="shareButton" 
            onClick={submitHandler} 
            type="submit"
            disabled={loading}
            style={{ backgroundColor: loading ? "gray" : "#1872f2"}}
          >
            {loading ? "Verifying..." : "Share"}
          </button>
        </form>
      </div>
    </ShareContainer>
  );
}

const ShareContainer = styled.div`
  width: 100%;
  border-radius: 10px;
  -webkit-box-shadow: 0px 0px 16px -8px rgba(0, 0, 0, 0.68);
  box-shadow: 0px 0px 16px -8px rgba(0, 0, 0, 0.68);
  .shareWrapper {
    padding: 10px;
    margin: 10px;
  }
  .shareTop {
    display: flex;
    align-items: center;
  }
  .shareInput {
    border: none;
    width: 100%;
  }
  .shareInput:focus {
    outline: none;
  }
  .shareHr {
    margin: 20px;
  }
  .shareBottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .shareOptions {
    display: flex;
    margin-left: 20px;
  }
  .shareOption {
    display: flex;
    align-items: center;
    margin-right: 15px;
    cursor: pointer;
  }
  .shareIcon {
    font-size: 18px;
    margin-right: 3px;
  }
  .shareOptionText {
    font-size: 14px;
    font-weight: 500;
  }
  .shareButton {
    border: none;
    padding: 7px;
    border-radius: 5px;
    background-color: #1872f2;
    font-weight: 500;
    margin-right: 20px;
    cursor: pointer;
    color: white;
  }
`;

export default Share;