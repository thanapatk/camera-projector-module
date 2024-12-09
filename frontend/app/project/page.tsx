"use client";

import { useState, useRef, useEffect, useContext } from "react";
import {
    ArrowLeft,
    RotateCcw,
    Maximize2,
    Circle,
    RectangleHorizontal,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { WebsocketContext } from "@/components/WebSocket";

interface Position {
    x: number;
    y: number;
}

type ShapeType = "rectangle" | "circle";

export default function ImagePositionPage() {
    const [ready, val, send] = useContext(WebsocketContext);

    const [imagePosition, setImagePosition] = useState<Position>({
        x: 50,
        y: 50,
    });
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState(false);
    const [dragStart, setDragStart] = useState<Position>({ x: 0, y: 0 });
    const [scale, setScale] = useState(1);
    const [initialScale, setInitialScale] = useState(1);
    const [resizeStartPos, setResizeStartPos] = useState<Position>({
        x: 0,
        y: 0,
    });
    const [selectedShape, setSelectedShape] = useState<ShapeType>("rectangle");

    const [projectorReady, setProjectorReady] = useState<boolean>(false);
    const [aspectRatio, setAspectRatio] = useState(1);

    const containerRef = useRef<HTMLDivElement>(null);
    const overlayRef = useRef<HTMLDivElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);
    const router = useRouter();

    useEffect(() => {
        if (!ready) return;
        console.log(val);
        if (val == null) return;

        const cmd = val.split(" ")[0];

        if (cmd === "detecting_object") setProjectorReady(false);
        else if (cmd === "detected_ratio") {
            const ratio = parseFloat(val.split(" ")[1]);
            setAspectRatio(ratio);
            setProjectorReady(true);
        }
    }, [ready, val]);

    // Calculate shape dimensions based on default width and aspect ratio
    const DEFAULT_WIDTH = 500;
    const shapeHeight = DEFAULT_WIDTH / aspectRatio;

    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDragging(true);
        setDragStart({
            x: e.clientX - imagePosition.x,
            y: e.clientY - imagePosition.y,
        });
    };

    const handleCornerMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsResizing(true);
        setResizeStartPos({ x: e.clientX, y: e.clientY });
        setInitialScale(scale);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging) {
            const newX = e.clientX - dragStart.x;
            const newY = e.clientY - dragStart.y;
            setImagePosition({ x: newX, y: newY });
        }

        if (isResizing && containerRef.current) {
            const dx = e.clientX - resizeStartPos.x;
            const dy = e.clientY - resizeStartPos.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const scaleFactor = dx > 0 ? 1 : -1;
            const scaleChange = (distance / 200) * scaleFactor;
            const newScale = Math.max(0.1, Math.min(3, initialScale + scaleChange));
            setScale(newScale);
        }
    };

    const handleProjectButton = (e: React.MouseEvent) => {
        e.preventDefault();

        const overlay = overlayRef.current;
        const image = imageRef.current;
        if (!overlay || !image) return;

        const overlayRect = overlay.getBoundingClientRect();
        const imageRect = image.getBoundingClientRect();
        const tx = imageRect.x - overlayRect.x;
        const ty = imageRect.y - overlayRect.y;

        console.log(
            `project ${tx} ${ty} ${imageRect.width / overlayRect.width} ${imageRect.height / overlayRect.height}`,
        );
        send(
            `project ${tx / overlayRect.width} ${ty / overlayRect.width} ${imageRect.width / overlayRect.width} ${imageRect.height / overlayRect.height}`,
        );
    };

    const handleMouseUp = () => {
        setIsDragging(false);
        setIsResizing(false);
    };

    const toggleShape = () => {
        setSelectedShape((prev) => (prev === "rectangle" ? "circle" : "rectangle"));
    };

    useEffect(() => {
        const handleGlobalMouseUp = () => {
            setIsDragging(false);
            setIsResizing(false);
        };
        window.addEventListener("mouseup", handleGlobalMouseUp);
        return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
    }, []);

    return (
        <div className="h-screen bg-black flex flex-col">
            <div className="flex-1 p-8">
                <div className="bg-gray-600 rounded-3xl p-8 max-w-5xl mx-auto h-full">
                    <div className="flex justify-center items-center gap-4 mb-4">
                        <button
                            onClick={toggleShape}
                            className="text-white flex items-center gap-2 bg-gray-700 px-4 py-2 rounded-lg"
                        >
                            {selectedShape === "rectangle" ? (
                                <>
                                    <RectangleHorizontal className="w-5 h-5" />
                                    Rectangle
                                </>
                            ) : (
                                <>
                                    <Circle className="w-5 h-5" />
                                    Circle
                                </>
                            )}
                        </button>
                    </div>

                    <h2 className="text-white text-xl text-center mb-8">
                        Drag to adjust position, use handle to resize
                    </h2>

                    <div
                        ref={containerRef}
                        className="relative bg-gray-600 rounded-lg overflow-hidden h-[500px]"
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                    >
                        {/* The draggable image container */}
                        <div
                            className="absolute cursor-move select-none"
                            style={{
                                transform: `translate(${imagePosition.x}px, ${imagePosition.y}px) scale(${scale})`,
                                transition:
                                    isDragging || isResizing ? "none" : "transform 0.1s ease-out",
                                //transformOrigin: "top left",
                            }}
                            onMouseDown={handleMouseDown}
                            onDragStart={(e) => e.preventDefault()}
                        >
                            <img
                                ref={imageRef}
                                src={
                                    typeof window !== "undefined"
                                        ? window.localStorage.getItem("uploadedImage")!
                                        : undefined
                                }
                                alt="Uploaded image"
                                className="w-full h-full object-contain pointer-events-none"
                                draggable="false"
                            />

                            {/* Single resize handle */}
                            <div
                                style={{
                                    transform: `scale(${1 / scale})`,
                                    transformOrigin: "center",
                                }}
                                className="absolute w-8 h-8 bg-white rounded-full cursor-nw-resize -right-4 -top-4 border-2 border-gray-800 flex items-center justify-center hover:bg-gray-100 transition-colors"
                                onMouseDown={handleCornerMouseDown}
                            >
                                <Maximize2 className="w-4 h-4 text-gray-800" />
                            </div>
                        </div>

                        {/* Shape overlay */}
                        {projectorReady ? (
                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                {selectedShape === "rectangle" ? (
                                    <div
                                        ref={overlayRef}
                                        className="border border-white"
                                        style={{
                                            width: DEFAULT_WIDTH,
                                            height: shapeHeight,
                                        }}
                                    />
                                ) : (
                                    <div
                                        ref={overlayRef}
                                        className="border border-white rounded-full"
                                        style={{
                                            width: DEFAULT_WIDTH,
                                            height:
                                                DEFAULT_WIDTH / (aspectRatio > 1 ? aspectRatio : 1),
                                        }}
                                    />
                                )}
                            </div>
                        ) : (
                            <></>
                        )}
                    </div>
                </div>
            </div>

            {/* Bottom toolbar */}
            <div className="bg-black p-6 flex items-center justify-between">
                <div className="flex items-center gap-8">
                    <button
                        onClick={() => router.back()}
                        className="flex items-center gap-2 text-white hover:text-gray-300 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span className="uppercase tracking-wider">Change Image</span>
                    </button>

                    <button
                        onClick={() => {
                            if (typeof window !== "undefined")
                                window.localStorage.removeItem("uploadedImage");
                            router.push("/");
                        }}
                        className="flex items-center gap-2 text-white hover:text-gray-300 transition-colors"
                    >
                        <RotateCcw className="w-5 h-5" />
                        <span className="uppercase tracking-wider">Start Over</span>
                    </button>
                </div>

                <button
                    disabled={!projectorReady}
                    onClick={handleProjectButton}
                    className="bg-blue-600 text-white px-24 py-3 rounded-xl hover:bg-blue-700 transition-colors uppercase tracking-wider"
                >
                    Project
                </button>
            </div>
        </div>
    );
}
