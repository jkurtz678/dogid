import { useState } from 'react'
import cooperImage from './assets/cooper.JPG'
import viteLogo from '/vite.svg'
import './App.css'
import Chart from './Chart.tsx'
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <h1>PupParser</h1>
      <p>Identify your dog breed</p>
      <Button onClick={() => setCount(count + 1)} style={{margin: `${30}px ${0}`}}>Upload Your Dog</Button>
      <Card style={{marginTop: 60 + 'px', padding: `${25}px 0`, minWidth: `${500}px`}}>
        <CardContent style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
          <img src={cooperImage} alt="Vite logo"  style={{maxWidth: 250 + 'px'}}/>
          <Chart />
        </CardContent>
      </Card>
    </>
  )
}

export default App
