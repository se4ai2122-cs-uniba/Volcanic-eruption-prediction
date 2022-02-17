import React from 'react'
import axios from 'axios'
import {
    VStack,
    Button,
    OrderedList,
    ListItem,
   
} from '@chakra-ui/react'

const Endpoint = 'https://backend-qdc5brb42q-ew.a.run.app/'
const ModelsContext = React.createContext([])

function Models() {
    const [data, setModels] = React.useState([])
    const updateModels = async () => {
        await axios({
            url: Endpoint + 'models',
            method: 'get',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PATCH, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token'}      
        })
        .then(function(response) {
            setModels(response.data.data)
            console.log(data)
        })
        .catch(function(error) {
            console.log(error)
        })
    }

    return (
        <ModelsContext.Provider value={{data}}>
            <VStack>
                <Button
                    size="md"
                    colorScheme='orange'
                    variant='solid'
                    onClick={updateModels}>
                    Get Models List
                </Button>
                <OrderedList>
                {
                   data.map((name) => (
                    <ListItem key={name}>{JSON.stringify(name.type)+" -> RMSE: "+JSON.stringify(name.rmse)}</ListItem>
                ))           
                }
                </OrderedList>
            </VStack>
        </ModelsContext.Provider>
    )
}

export default Models