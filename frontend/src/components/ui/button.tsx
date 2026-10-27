import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-xs font-semibold transition-all focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-amber-400 text-slate-950 shadow hover:bg-amber-300 active:scale-[0.98]",
        outline:
          "border border-slate-700 bg-slate-900/90 text-slate-200 hover:border-amber-400/60 hover:text-amber-300",
        emerald:
          "bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/30",
        rose:
          "bg-rose-500/20 border border-rose-500/40 text-rose-300 hover:bg-rose-500/30",
        ghost: "text-slate-400 hover:bg-slate-800/60 hover:text-slate-100",
      },
      size: {
        default: "h-9 px-3.5 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-lg px-5 text-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
