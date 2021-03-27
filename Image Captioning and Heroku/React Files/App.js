import React from 'react'
import {StyleSheet, Text, View, Linking, Image} from 'react-native'
import { Helmet } from 'react-helmet'
// Import Bootstrap
 
const link_to_sample = 'https://github.com/ShaoA182739081729371028392/Machine-and-Deep-Learning-Mini-Projects/tree/main/Image%20Captioning%20and%20Heroku/Sample%20Images';


class App extends React.Component{
  
  onClick(event){
    if (event !== null){
      Linking.openURL(link_to_sample);
    }
  }
  
  onFileUpload(event){
    var file = event.target.files[0];
    var form = new FormData();
    form.append("file", file);
    let output = "";
    let fr = new FileReader();
    fr.onload = (e) => {
      fetch("http://localhost:5000/process", {method: "POST",
      body: form, 
      }).then((val) => {
        val.json().then((val2) => {
          output = val2;
          let selected = document.getElementById("image")
          
          selected.innerHTML = `<img src = ${e.target.result} height = 200 width = 200/>`
          var x= document.getElementById('Output')
          x.innerHTML = output
        })
      })
    }
    fr.readAsDataURL(file)
    

    
  }
  
  render(){
    return (
      <View style= {stylesheet.container}>
        <Helmet>
          <title>Image Captioning!</title>
        </Helmet>
          
        <Text style = {stylesheet.header}>
          Image Captioning System
        </Text>
        <Text style = {stylesheet.smallheader}>
          An Image Captioning System Constructed using a LSTM and ResNet200D BackBone.
        </Text>
        <span>&nbsp;</span>
        <span>&nbsp;</span>
        <Text style = {stylesheet.text}>
          This project was created using PyTorch, leveraging PyTorch Image Models. Created By: Andrew Shao.
        </Text>
        <span>&nbsp;</span>
        <Text style = {stylesheet.text} onClick = {this.onClick}>
          If you don't have an image to try, feel free to download one<span>&nbsp;</span>
          <Text style = {stylesheet.link}>
             here
          </Text>
        </Text>
        <span>&nbsp;</span>
        <Text style= {stylesheet.text}>
          Upload your Image Here(Accepted File Formats: JPG, PNG):
        </Text>
        <span>&nbsp;</span>
        <input type = 'file' name = 'file' onChange = {this.onFileUpload} accept = '.jpg, .jpeg, .png'></input>
        <span>&nbsp;</span>
        <Text nativeID = "image"></Text>
        <Text style = {stylesheet.text} nativeID = "Output"></Text>
      </View>
    )
  }
}
const stylesheet = StyleSheet.create({
  container: {
    padding: 50,
    flex: 1
  },
  header: {
    fontFamily: "Tahoma",
    color: "orange",
    fontWeight: "bold",
    fontSize: 36
  },
  link: {
    fontFamily: "Tahoma",
    color: 'blue',
    textDecorationLine: "underline",
    fontSize: 20
  },
  smallheader: {
    fontFamily: "Tahoma",
    color: 'black',
    fontWeight: 'bold', 
    fontSize: 24
  },
  text: {
    fontSize: 20,
    fontFamily: "Tahoma",
    color: "black"
  }
})
export default App;
