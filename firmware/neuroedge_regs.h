#pragma once
#include <stdint.h>
#define NEURO_BASE          0x40000000u
#define NEURO_CONTROL       (*(volatile uint32_t*)(NEURO_BASE + 0x00))
#define NEURO_THRESHOLD     (*(volatile uint32_t*)(NEURO_BASE + 0x04))
#define NEURO_STATUS        (*(volatile uint32_t*)(NEURO_BASE + 0x08))
#define NEURO_EVENT_COUNT   (*(volatile uint32_t*)(NEURO_BASE + 0x0C))
#define NEURO_CYCLES        (*(volatile uint32_t*)(NEURO_BASE + 0x10))
#define NEURO_CONTROL_START (1u << 0)
#define NEURO_CONTROL_EVENT (1u << 1)
