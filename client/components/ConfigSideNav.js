"use client";
import React, { useState } from "react";
import { Button, Stack, Form, Spinner, ToggleButtonGroup } from "react-bootstrap";
import { ToastContainer, toast } from "react-toastify";

export default function ConfigSideNav() {
  const [downloadInProgress, setdownloadInProgress] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelName, setModelName] = useState('GPT4All');

  const ingestData = async () => {
    try {
      setIsLoading(true);
      const res = await fetch("http://localhost:5000/ingest");
      const jsonData = await res.json();
      if (!res.ok) {
        // This will activate the closest `error.js` Error Boundary
        console.log("Error Ingesting data");
        setIsLoading(false);
      } else {
        setIsLoading(false);
        console.log(jsonData);
      }
    } catch (error) {
      setIsLoading(false);
	  response.text().then(text => {toast.error("Error Ingesting data."+text);})
    }
  };

  const handleDownloadModel = async () => {
    try {
      setdownloadInProgress(true);
      const res = await fetch(`http://localhost:5000/download_model?model_name=${modelName}`);
      const jsonData = await res.json();
      if (!res.ok) {
	    response.text().then(text => {toast.error("Error downloading model."+text);})  
        setdownloadInProgress(false);
      } else {
        setdownloadInProgress(false);
        toast.success("Model Download complete");
        console.log(jsonData);
      }
    } catch (error) {
      setdownloadInProgress(false);
      console.log(error);
      toast.error("Error downloading model");
    }
  };

  const handleFileChange = (event) => {
    if(event.target.files[0]!=null){
      setSelectedFile(event.target.files[0]);
    }
    
  };

  const handleUpload = async () => {
    setIsUploading(true)
    try {
      const formData = new FormData();
      formData.append("document", selectedFile);

      const res = await fetch("http://localhost:5000/upload_doc", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        console.log("Error Uploading document");
		response.text().then(text => {toast.error("Error Uploading document."+text);})
        setSelectedFile(null); // Clear the selected file after successful upload
        document.getElementById("file-input").value = "";
        setIsUploading(false)
      } else {
        const data = await res.json();
        console.log(data);
        toast.success("Document Upload Successful");
        setSelectedFile(null); // Clear the selected file after successful upload
        document.getElementById("file-input").value = "";
        setIsUploading(false)
      }
    } catch (error) {
      console.log("error");
      toast.error("Error Uploading document");
      setSelectedFile(null); // Clear the selected file after successful upload
      document.getElementById("file-input").value = "";
      setIsUploading(false)
    }
  };

  return (
    <>
      <div className="mx-4 mt-3">
        <Form>
          <Form.Check
            type='radio'
            label={'Falcon-7B'}
            id={'Falcon-7B'}
            value={'tiiuae/falcon-7b'}
            checked={modelName==='tiiuae/falcon-7b'}
            onChange={(e) => {
              setModelName(e.target.value);
            }}
          />
          <Form.Check
            type='radio'
            label={'Koala 13b'}
            id={'Koala 13b'}
            value={'TheBloke/koala-13B-HF'}
            checked={modelName==='TheBloke/koala-13B-HF'}
            onChange={(e) => {
              setModelName(e.target.value);
            }}
          />
          <Form.Check
            type='radio'
            label={'Vicuna 13b'}
            id={'Vicuna 13b'}
            value={'lmsys/vicuna-13b-delta-v0'}
            checked={modelName==='lmsys/vicuna-13b-delta-v0'}
            onChange={(e) => {
              setModelName(e.target.value);
            }}
          />
          <Form.Check
            type='radio'
            label={'GPT4All'}
            id={'GPT4All'}
            value={'nomic-ai/gpt4all-j'}
            checked={modelName==='nomic-ai/gpt4all-j'}
            onChange={(e) => {
              setModelName(e.target.value);
            }}
          />
        </Form>
      </div>
      <Stack direction="horizontal" className="mx-4 mt-5" gap={3}>
        {downloadInProgress ? (
          <div className="d-flex justify-content-center"><Spinner animation="border" /><span className="ms-3">downloading</span></div>
        ) : (
          <div>
            <Button
              onClick={(e) => {
                handleDownloadModel();
              }}
            >
              Download Model
            </Button>
          </div>
        )}
      </Stack>
    </>
  );
}
