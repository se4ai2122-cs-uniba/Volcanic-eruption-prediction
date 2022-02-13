import React from 'react'
import axios from 'axios'
import {
    VStack,
    Button,
    OrderedList,
    ListItem,
   
} from '@chakra-ui/react'

const Endpoint = 'http://localhost:5000/'
const ModelsContext = React.createContext([])

function Models() {
    const [data, setModels] = React.useState([])
    const updateModels = async () => {
        await axios({
            url: Endpoint + 'models',
            method: 'get',
            headers: {
                Accept: 'application/json'
            }
        })
        .then(function(response) {
            setModels(response.data.data)
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