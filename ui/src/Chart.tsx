"use client"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis, ResponsiveContainer } from "recharts"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

const chartData = [
  { breed: "Chow", percentage: 44 },
  { breed: "Golden Retriever", percentage: 25},
  { breed: "Husky", percentage: 19},
  { breed: "Labrador Retriever", percentage: 5 },
  { breed: "Other", percentage: 7 },
]

const chartConfig = {
  percentage: {
    label: "Percentage",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

export default function Component() {
  return (
        <div className="h-[250px] w-full" style={{marginTop: `${25}px`}}>
          <ChartContainer config={chartConfig}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{
                  right: 10,
                  left: 10
                }}
                barSize={30}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis 
                  type="number" 
                  domain={[0, 50]}
                  hide
                />
                <YAxis
                  dataKey="breed"
                  type="category"
                  width={100}
                  tickMargin={10}
                  axisLine={false}
                  tick={{ fontSize: 12 }}
                />
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent indicator="line" />}
                />
                <Bar
                  dataKey="percentage"
                  fill="var(--color-desktop)"
                  radius={4}
                  label={({ x, y, width, value }) => (
                    <text
                      x={x + width + 5}
                      y={y + 15}
                      fill="currentColor"
                      className="fill-foreground text-sm"
                      dominantBaseline="middle"
                    >
                      {value}%
                    </text>
                  )}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
  )
}
