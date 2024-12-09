import { WebsocketProvider } from "@/components/WebSocket";

export default function YogaLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <WebsocketProvider url="ws://thanapatk.local:8000/project_ws">
            {children}
        </WebsocketProvider>
    );
}
