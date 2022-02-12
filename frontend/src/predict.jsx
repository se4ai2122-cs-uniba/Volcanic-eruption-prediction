import React from 'react'
import axios from 'axios'
import {
    VStack,
    Button,
    Center,
    Input,
    Heading
} from '@chakra-ui/react'

const Endpoint = 'http://localhost:5000/predict/'
const PredictContext = React.createContext(null)

function Predict() {
    const [file, setFile] = React.useState(null)
    const [prediction, setPrediction] = React.useState(null)
    const [message, setMessage] = React.useState(null)
    const handleSubmit = (e) => {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const type = formData.get('type');
        formData.delete('type');
        axios.post(Endpoint + type, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            params: {
                type
            },
        }).then(response => {
            setPrediction(response.data.data)
            setFile(response.data)
            setMessage(response.data)
            console.log(response.data)
            if((response.data.message)==("OK")){
                setMessage(response.data.message="")}
        })
    }

    return (
        <VStack>
            <form onSubmit={handleSubmit}>
                <Input placeholder='Insert type of model' name="type" />
                <br />  
                <br />
                <Input type='file' name="file"/>
                <br />
                <br />
                <Center>
                    <Button
                        type='submit'
                        size="md"
                        colorScheme='orange'
                        variant='solid'>
                        Make Prediction
                    </Button>
                </Center>
            </form>
            <br />
            <br />
            <PredictContext.Provider value={{message}}>
                {
                    message ?
                        <Heading as='h4' size='md' color='red.700' >{JSON.stringify(message.message)}</Heading>
                    :
                        null
                }
            </PredictContext.Provider>
            <PredictContext.Provider value={{prediction}}>
                {
                    prediction ?
                        <Heading as='h4' size='md' color='red.700' >{"Next volcano's eruption predicted in: " + JSON.stringify(prediction.prediction.eruption_time)}</Heading>
                    :
                        null
                }
            </PredictContext.Provider>
        </VStack>
    ) 
}

export default Predict