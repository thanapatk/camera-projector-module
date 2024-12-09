"use client";

import { createContext, useState, useRef, useEffect, ReactNode } from "react";

export const WebsocketContext = createContext<
    [boolean, any, WebSocket["send"]]
>([false, null, () => { }]); // ready, value, send

// Make sure to put WebsocketProvider higher up in
// the component tree than any consumer.
export const WebsocketProvider = ({
    children,
    url,
}: {
    children: ReactNode;
    url: string;
}) => {
    const [isReady, setIsReady] = useState(false);
    const [val, setVal] = useState<any>(null);

    const ws = useRef<WebSocket | null>(null);

    useEffect(() => {
        const socket = new WebSocket(url);

        socket.onopen = () => setIsReady(true);
        socket.onclose = () => setIsReady(false);
        socket.onmessage = (event) => setVal(event.data);

        ws.current = socket;

        return () => {
            if (socket.readyState === 1) socket.close();
        };
    }, [url]);

    const ret = [isReady, val, ws.current?.send.bind(ws.current)];

    return (
        <WebsocketContext.Provider value={ret as [boolean, any, WebSocket["send"]]}>
            {children}
        </WebsocketContext.Provider>
    );
};
