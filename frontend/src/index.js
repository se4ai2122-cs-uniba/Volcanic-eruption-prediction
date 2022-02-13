import React from 'react';
import ReactDOM from 'react-dom';




import Header from './header'
import Models from './models'
import Predict from './predict'

import {
  ChakraProvider,
  Center,
  VStack,
  Divider
} from '@chakra-ui/react'

function App({ Component }) {
  return (
    <ChakraProvider>
      <Header />
      <Center bg='white' padding='5mm'>
          <VStack divider={<Divider />} >
              <Models />
              <Predict />
          </VStack>
      </Center>
    </ChakraProvider>
  )
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)