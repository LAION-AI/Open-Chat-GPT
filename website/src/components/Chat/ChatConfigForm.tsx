import {
  Box,
  Button,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Input,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Select,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Stack,
  Switch,
  Tooltip,
  useBoolean,
  useToast,
} from "@chakra-ui/react";
import { Check, X } from "lucide-react";
import { useTranslation } from "next-i18next";
import { ChangeEvent, KeyboardEvent, memo, useCallback, useEffect, useRef, useState } from "react";
import { Controller, useFormContext, UseFormSetValue } from "react-hook-form";
import SimpleBar from "simplebar-react";
import { ChatConfigFormData, ModelParameterConfig, PluginEntry, SamplingParameters } from "src/types/Chat";
import { CustomPreset, getConfigCache } from "src/utils/chat";
import { useIsomorphicLayoutEffect } from "usehooks-ts";

import { ChatConfigSaver } from "./ChatConfigSaver";
import { useChatInitialData } from "./ChatInitialDataContext";
import { PluginsChooser } from "./PluginsChooser";
import { areParametersEqual } from "./WorkParameters";

const sliderItems: Readonly<
  Array<{
    key: keyof SamplingParameters;
    max?: number;
    min?: number;
    precision?: number;
    step?: number;
  }>
> = [
  {
    key: "temperature",
    min: 0.01,
    max: 2,
  },
  {
    key: "max_new_tokens",
    max: 1024,
    step: 1,
    min: 1,
  },
  {
    key: "top_p",
  },
  {
    key: "repetition_penalty",
    min: 1,
    max: 3,
  },
  {
    key: "top_k",
    min: 5,
    max: 2000,
    step: 5,
  },
  {
    key: "typical_p",
  },
];

const unKnownCustomPresetName = "__custom__";
const customPresetNamePrefix = "$$";

const parameterLabel: Record<keyof SamplingParameters, string> = {
  max_new_tokens: "Max new tokens",
  top_k: "Top K",
  top_p: "Top P",
  temperature: "Temperature",
  repetition_penalty: "Repetition Penalty",
  typical_p: "Typical P",
};

const findPresetName = (presets: ModelParameterConfig[], config: SamplingParameters) => {
  return (
    presets.find((preset) => areParametersEqual(preset.sampling_parameters, config))?.name ?? unKnownCustomPresetName
  );
};

const resetParameters = (setValue: UseFormSetValue<ChatConfigFormData>, params: SamplingParameters) => {
  for (const [key, value] of Object.entries(params) as Array<[keyof SamplingParameters, number]>) {
    setValue(key, value); // call setValue instead of setValues to avoid reset unwanted fields
  }
};

export const ChatConfigForm = memo(function ChatConfigForm() {
  const { t } = useTranslation("chat");
  const { modelInfos } = useChatInitialData();

  const { control, getValues, register, setValue } = useFormContext<ChatConfigFormData>();
  const selectedModel = getValues("model_config_name"); // have to use getValues to here to access latest value
  const selectedPlugins = getValues("plugins");
  const presets = modelInfos.find((model) => model.name === selectedModel)!.parameter_configs;
  const [selectedPresetName, setSelectedPresetName] = useState(() => findPresetName(presets, getValues()));
  const { hyrated, plugins, setPlugins, customPresets, setCustomPresets } = useHydrateChatConfig({
    setSelectedPresetName,
  });

  const [lockPresetSelection, setLockPresetSelection] = useState(false);

  const handlePresetChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newPresetName = e.target.value;
      if (newPresetName !== unKnownCustomPresetName) {
        const config = presets.find((preset) => preset.name === newPresetName)!.sampling_parameters;
        resetParameters(setValue, config);
      }
      setSelectedPresetName(newPresetName);
    },
    [presets, setValue]
  );

  // Lock preset selection if any plugin is enabled
  useEffect(() => {
    const activated = selectedPlugins.some((plugin) => plugin.enabled);
    if (activated) {
      handlePresetChange({ target: { value: "k50-Plugins" } } as any);
      setLockPresetSelection(true);
    } else {
      setLockPresetSelection(false);
    }
  }, [presets, selectedPlugins, handlePresetChange, getValues]);

  const handleSavePreset = useCallback(
    (name: string) => {
      const prefixedName = `${customPresetNamePrefix}${name}`;
      setCustomPresets((prev) => [...prev, { name: prefixedName, config: getValues() }]);
      setSelectedPresetName(prefixedName);
    },
    [getValues, setCustomPresets]
  );

  return (
    <>
      <SimpleBar
        style={{ maxHeight: "100%", height: "100%", minHeight: "0" }}
        classNames={{
          contentEl: "mr-4 flex flex-col overflow-y-auto items-center",
        }}
      >
        <Stack gap="4" maxW="full">
          <PluginsChooser plugins={plugins} setPlugins={setPlugins} />
          <FormControl>
            <FormLabel>{t("model")}</FormLabel>
            <Select {...register("model_config_name")}>
              {modelInfos.map(({ name }) => (
                <option value={name} key={name}>
                  {name}
                </option>
              ))}
            </Select>
          </FormControl>
          <FormControl>
            <FormLabel>{t("preset")}</FormLabel>
            <Select value={selectedPresetName} onChange={handlePresetChange} isDisabled={lockPresetSelection}>
              {presets.map(({ name }) => (
                <option value={name} key={name}>
                  {name}
                </option>
              ))}
              {customPresets.map(({ name }) => (
                <option value={name} key={name}>
                  {name.slice(customPresetNamePrefix.length)}
                </option>
              ))}
              <option value={unKnownCustomPresetName}>{t("preset_custom")}</option>
            </Select>
          </FormControl>
          {sliderItems.map((item) => (
            <Controller
              name={item.key}
              key={item.key}
              control={control}
              render={({ field: { onChange, name } }) => (
                <ChatParameterField
                  {...item}
                  value={getValues(name)} // need to call getValues here, react-hook-form not trigger rerender when call setValue manually
                  onChange={onChange}
                  name={name}
                  isDisabled={
                    selectedPresetName !== unKnownCustomPresetName &&
                    !selectedPresetName.startsWith(customPresetNamePrefix)
                  }
                  description={t(("parameter_description." + name) as any)}
                />
              )}
            ></Controller>
          ))}
        </Stack>
        <ChatConfigSaver
          plugins={plugins}
          hyrated={hyrated}
          selectedPresetName={selectedPresetName}
          customPresets={customPresets}
        />
      </SimpleBar>
      {selectedPresetName === unKnownCustomPresetName && (
        <SavePresetButton customPresets={customPresets} onSave={handleSavePreset} />
      )}
    </>
  );
});

const useHydrateChatConfig = ({ setSelectedPresetName }: { setSelectedPresetName: (preset: string) => void }) => {
  const { modelInfos, builtInPlugins } = useChatInitialData();
  const hyrated = useRef(false);
  const { setValue } = useFormContext<ChatConfigFormData>();
  const [plugins, setPlugins] = useState<PluginEntry[]>(builtInPlugins);
  const [customPresets, setCustomPresets] = useState<CustomPreset[]>([]);

  useIsomorphicLayoutEffect(() => {
    if (hyrated.current) return;

    hyrated.current = true;
    const cache = getConfigCache();

    if (!cache) {
      return;
    }

    const { selectedPresetName, model_config_name, custom_preset_config, selectedPlugins, plugins, custom_presets } =
      cache;
    const model = modelInfos.find((model) => model.name === model_config_name);

    if (model) {
      setValue("model_config_name", model_config_name);
    }

    if (plugins) {
      // filter out duplicated with built-in plugins and dedup by url
      const dedupedCustomPlugins = [
        ...new Map(
          plugins
            .filter((plugin) => builtInPlugins.findIndex((p) => p.url === plugin.url) === -1)
            .map((item) => [item.url, item])
        ).values(),
      ];
      setPlugins([...builtInPlugins, ...dedupedCustomPlugins]);
    }

    if (custom_presets) {
      setCustomPresets(custom_presets);
    }

    if (selectedPlugins && selectedPlugins.length > 0) {
      setValue("plugins", selectedPlugins);
      const preset = (model || modelInfos[0]).parameter_configs.find(
        (preset) => preset.name === "k50-Plugins"
      )?.sampling_parameters;
      if (preset) {
        resetParameters(setValue, preset);
      }
    } else {
      // only hydrate sampling params if there is no selected plugins
      if (selectedPresetName === unKnownCustomPresetName) {
        resetParameters(setValue, custom_preset_config);
        setSelectedPresetName(selectedPresetName);
      } else if (selectedPresetName.startsWith(customPresetNamePrefix)) {
        const customPreset = customPresets.find((preset) => preset.name === selectedPresetName)?.config;
        if (customPreset) {
          resetParameters(setValue, customPreset);
          setSelectedPresetName(selectedPresetName);
        }
      } else {
        // built-in preset
        const preset = (model || modelInfos[0]).parameter_configs.find(
          (preset) => preset.name === selectedPresetName
        )?.sampling_parameters;
        if (preset) {
          resetParameters(setValue, preset);
          setSelectedPresetName(selectedPresetName);
        }
      }
    }
  }, [modelInfos]);

  return { hyrated, plugins, setPlugins, customPresets, setCustomPresets };
};

type NumberInputSliderProps = {
  max?: number;
  min?: number;
  precision?: number;
  step?: number;
  onChange: (value: number | null) => void;
  value: number | null;
  isDisabled: boolean;
  name: keyof SamplingParameters;
  description?: string;
};

const ChatParameterField = memo(function ChatParameterField(props: NumberInputSliderProps) {
  const { max = 1, precision = 2, step = 0.01, min = 0, value, isDisabled, description, name, onChange } = props;

  const handleChange = useCallback(
    (val: string | number) => {
      onChange(Number(val));
    },
    [onChange]
  );

  const handleShowSliderChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const checked = e.target.checked;
      onChange(checked ? max : null);
    },
    [onChange, max]
  );
  const label = parameterLabel[name];
  const showSlider = value !== null;

  return (
    <FormControl isDisabled={isDisabled}>
      <Flex justifyContent="space-between" mb="2">
        <FormLabel mb="0">
          <Tooltip label={description} placement="left">
            {label}
          </Tooltip>
        </FormLabel>
        <Switch isChecked={showSlider} onChange={handleShowSliderChange}></Switch>
      </Flex>
      {showSlider && (
        <Flex gap="4">
          <Slider
            aria-label={label}
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={handleChange}
            focusThumbOnChange={false}
            isDisabled={isDisabled}
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
          </Slider>
          <NumberInput
            value={value}
            onChange={handleChange}
            size="xs"
            maxW="80px"
            precision={precision}
            step={step}
            min={min}
            max={max}
            isDisabled={isDisabled}
          >
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper height="12px" />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </Flex>
      )}
    </FormControl>
  );
});

const SavePresetButton = ({
  customPresets,
  onSave,
}: {
  customPresets: CustomPreset[];
  onSave: (name: string) => void;
}) => {
  const [isSaving, setIsSaving] = useBoolean();
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const handleSave = useCallback(() => {
    const name = inputRef.current?.value.trim();
    if (!name) {
      return;
    }

    const isExists = customPresets.findIndex((preset) => preset.name === name) !== -1;

    if (isExists) {
      toast({
        title: t("chat:preset_exists"),
        status: "error",
      });
    } else {
      onSave(name);
      setIsSaving.off();
    }
  }, [customPresets, onSave, setIsSaving, t, toast]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsSaving.off();
      }
      if (e.key === "Enter") {
        handleSave();
      }
    },
    [handleSave, setIsSaving]
  );

  return (
    <Box pe="4" position="relative">
      {!isSaving ? (
        <Button onClick={setIsSaving.on} py="3" variant="outline" colorScheme="blue" w="full">
          {t("chat:save_preset")}
        </Button>
      ) : (
        <>
          <Input
            py="3"
            pe="56px"
            ref={inputRef}
            onKeyDown={handleKeyDown}
            autoFocus
            placeholder={t("chat:preset_name_placeholder")}
          ></Input>
          <Flex position="absolute" top="2" className="ltr:right-6 rtl:left-6" gap="1" zIndex="10">
            <IconButton
              size="xs"
              variant="ghost"
              icon={<Check size="16px" />}
              aria-label={t("save")}
              onClick={handleSave}
            ></IconButton>
            <IconButton
              size="xs"
              variant="ghost"
              icon={<X size="16px" />}
              aria-label={t("cancel")}
              onClick={setIsSaving.off}
            ></IconButton>
          </Flex>
        </>
      )}
    </Box>
  );
};
