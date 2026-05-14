interface Props {
  label: string;
  probability: number;
}

export const PredictionBar = ({ label, probability }: Props) => {
  const color =
    probability > 0.7 ? "var(--severity-high)" :
    probability > 0.4 ? "var(--severity-medium)" :
                        "var(--severity-low)";
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 13, marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 500, color }}>{(probability * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "var(--surface-3)", borderRadius: 3 }}>
        <div style={{
          height: "100%", width: `${probability * 100}%`,
          background: color, borderRadius: 3, transition: "width 0.4s ease",
        }}/>
      </div>
    </div>
  );
};
