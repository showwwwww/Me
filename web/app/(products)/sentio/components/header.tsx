import React from "react";
import { LogoBar } from "@/components/header/logo";
import { Switch, addToast } from "@heroui/react";
import { ResourceModel } from "@/lib/protocol";
import { UserMinusIcon, UserPlusIcon } from "@heroicons/react/24/solid";
import { useSentioChatModeStore, useChatRecordStore, useSentioAsrStore, useSentioBackgroundStore, useSentioAgentStore } from "@/lib/store/sentio";
import { CHAT_MODE } from "@/lib/protocol";
import { useTranslations } from 'next-intl';

function DefaultEngine({ children }: { children: React.ReactNode }) {
    const { setSettings } = useSentioAgentStore();
    React.useEffect(() => {
        setSettings({ agent_type: 'local_lib', agent_module: 'adh_cv_agent.cv_rag_agent' });
    }, []);
    return children;
}

function DefaultBackground({ children }: { children: React.ReactNode }) {
    const { setBackground } = useSentioBackgroundStore();
    React.useEffect(() => {
        setBackground({
            link: '/sentio/backgrounds/static/夜晚街道.jpg',
            name: '夜晚街道',
            resource_id: 'STATIC_夜晚街道.jpg',
            sub_type: 'STATIC',
            type: 'background',
        } as ResourceModel);
    }, []);
    return children;
}

function ChatModeSwitch() {
    const t = useTranslations('Products.sentio');
    const { chatMode, setChatMode } = useSentioChatModeStore();
    const { enable } = useSentioAsrStore();
    const { clearChatRecord } = useChatRecordStore();
    const onSelect = (isSelected: boolean) => {
        if (enable) {
            setChatMode(isSelected ? CHAT_MODE.IMMSERSIVE : CHAT_MODE.DIALOGUE)
            clearChatRecord();   
        } else {
            addToast({
                title: t('asrEnableTip'),
                color: "warning"
            })
        }
    }
    return (
        <Switch
            color="secondary"
            startContent={<UserPlusIcon/>}
            endContent={<UserMinusIcon/>}
            isSelected={chatMode == CHAT_MODE.IMMSERSIVE}
            onValueChange={onSelect}
        />
    )
}

export function Header() {
    return (
        <DefaultBackground>
            <DefaultEngine>
                <div className="flex w-full h-[64px] p-6 justify-between z-10">
                    <LogoBar isExternal={true}/>
                    <div className="flex flex-row gap-4 items-center">
                        <ChatModeSwitch />
                    </div>
                </div>
            </DefaultEngine>
        </DefaultBackground>
    )
}