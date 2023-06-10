// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class G_520Test : ModuleRules
{
	public G_520Test(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay", "EnhancedInput", "OpenCV", "OpenCVHelper" });
		
		PrivateDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "Renderer", "RenderCore", "RHI", "RHICore", "D3D12RHI", "OpenCV", "OpenCVHelper" });
	}
}
