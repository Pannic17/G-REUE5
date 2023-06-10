// Copyright Epic Games, Inc. All Rights Reserved.

#include "G_520TestGameMode.h"
#include "G_520TestCharacter.h"
#include "UObject/ConstructorHelpers.h"

AG_520TestGameMode::AG_520TestGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
