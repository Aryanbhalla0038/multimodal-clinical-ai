import {
  ResponsiveContainer, ComposedChart, Bar, Line, XAxis, YAxis,
  Tooltip, Legend, CartesianGrid,
} from "recharts";

interface Props {
  importance: number[] | null;
}

export const VitalsChart = ({ importance }: Props) => {
  if (!importance || importance.length === 0) return null;
  const data = importance.map((v, i) => ({ hour: i, importance: v }));
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Vitals timestep importance</h3>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/>
          <XAxis dataKey="hour" label={{ value: "Hour", position: "insideBottom", offset: -2 }}/>
          <YAxis domain={[0, 1]}/>
          <Tooltip/>
          <Legend/>
          <Bar dataKey="importance" fill="var(--brand-primary)" fillOpacity={0.7}/>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};
