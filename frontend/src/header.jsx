import {
    Center,
    VStack,
    Divider,
    Heading,
    Text
} from '@chakra-ui/react'

const Header = () => {
    return (
        <Center bg='white' padding='5mm'>
            <VStack divider={<Divider />} >
                <Heading as='h1' size='2xl'>Vulcanic Prediction</Heading>
                <Text fontSize='2xl'>Make predictions on the next volcano's eruption</Text>
            </VStack>
        </Center>
    )
}

export default Header