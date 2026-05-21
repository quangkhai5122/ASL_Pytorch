import { useState } from "react";
import { useASL } from "@/context/ASLContext";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Camera, Sun, Square, ArrowRight } from "lucide-react";

const steps = [
  {
    icon: <Camera className="w-12 h-12 text-primary" aria-hidden="true" />,
    title: "Position Your Camera",
    description: "Place your camera at chest height, about arm's length away. Make sure your hands and upper body are visible in the frame.",
  },
  {
    icon: <Sun className="w-12 h-12 text-primary" aria-hidden="true" />,
    title: "Check Your Lighting",
    description: "Ensure even, front-facing lighting. Avoid backlighting or harsh shadows that could interfere with hand detection.",
  },
  {
    icon: <Square className="w-12 h-12 text-primary" aria-hidden="true" />,
    title: "Keep Background Simple",
    description: "A plain, uncluttered background helps the system better detect your hands and movements. Avoid busy patterns.",
  },
];

export function OnboardingModal() {
  const { onboardingOpen, setOnboardingOpen } = useASL();
  const [step, setStep] = useState(0);

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    } else {
      setOnboardingOpen(false);
    }
  };

  return (
    <Dialog open={onboardingOpen} onOpenChange={setOnboardingOpen}>
      <DialogContent className="sm:max-w-md" aria-describedby="onboarding-description">
        <p id="onboarding-description" className="sr-only">Getting started guide for ASL-Bridge</p>
        <div className="flex flex-col items-center text-center py-6 space-y-6">
          {/* Step indicator */}
          <div className="flex items-center gap-2" aria-label={`Step ${step + 1} of ${steps.length}`}>
            {steps.map((_, i) => (
              <div
                key={i}
                className={`w-2.5 h-2.5 rounded-full transition-colors ${
                  i === step ? "bg-primary" : i < step ? "bg-primary/40" : "bg-muted"
                }`}
              />
            ))}
          </div>

          {/* Content */}
          <div className="animate-fade-in" key={step}>
            {steps[step].icon}
            <h2 className="text-xl font-bold mt-4">{steps[step].title}</h2>
            <p className="text-sm text-muted-foreground mt-2 max-w-xs leading-relaxed">
              {steps[step].description}
            </p>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 w-full">
            <Button variant="ghost" className="touch-target flex-1" onClick={() => setOnboardingOpen(false)}>
              Skip
            </Button>
            <Button className="touch-target flex-1" onClick={handleNext}>
              {step < steps.length - 1 ? (
                <>Next <ArrowRight className="w-4 h-4 ml-1.5" aria-hidden="true" /></>
              ) : (
                "Get Started"
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
