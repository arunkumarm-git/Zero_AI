import { MoreVert } from "@material-ui/icons";
import styled from "styled-components";
import { format } from "timeago.js"; // Make sure to npm install timeago.js if you haven't

export default function Post({ post }) {
  return (
    <PostContainer>
      <div className="postWrapper">
        <div className="postTop">
          <div className="postTopLeft">
            <img
              className="postProfileImg"
              src="/images/defaultavatar.png" // Default avatar for now
              alt=""
            />
            <span className="postUsername">Arun (User)</span>
            <span className="postDate">{format(post.created_at)}</span>
          </div>
          <div className="postTopRight">
            <MoreVert />
          </div>
        </div>
        <div className="postCenter">
          <span className="postText">{post.description}</span>
          {/* Use the image_url from your backend */}
          <img className="postImg" src={post.image_url} alt="" />
        </div>
        <div className="postBottom">
          <div className="postBottomLeft">
            {/* Show Verification Badge */}
            {post.verified ? (
                <span style={{color: "green", fontWeight: "bold"}}>✅ Human Verified</span>
            ) : null}
          </div>
        </div>
      </div>
    </PostContainer>
  );
}

const PostContainer = styled.div`
  width: 100%;
  border-radius: 10px;
  -webkit-box-shadow: 0px 0px 16px -8px rgba(0, 0, 0, 0.68);
  box-shadow: 0px 0px 16px -8px rgba(0, 0, 0, 0.68);
  margin: 30px 0;

  .postWrapper {
    padding: 10px;
  }
  .postTop {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .postTopLeft {
    display: flex;
    align-items: center;
  }
  .postProfileImg {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 10px;
  }
  .postUsername {
    font-size: 15px;
    font-weight: 500;
    margin: 0 10px;
  }
  .postDate {
    font-size: 12px;
  }
  .postCenter {
    margin: 20px 0;
  }
  .postImg {
    margin-top: 20px;
    width: 100%;
    max-height: 500px;
    object-fit: contain;
  }
  .postBottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;