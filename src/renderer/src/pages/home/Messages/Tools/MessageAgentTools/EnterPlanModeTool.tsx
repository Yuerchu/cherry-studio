import type { CollapseProps } from 'antd'
import { useTranslation } from 'react-i18next'

import { ToolHeader } from './GenericTools'
import type { EnterPlanModeToolOutput } from './types'
import { AgentToolsType } from './types'

export function EnterPlanModeTool({
  output
}: {
  input?: Record<string, never>
  output?: EnterPlanModeToolOutput
}): NonNullable<CollapseProps['items']>[number] {
  const { t } = useTranslation()

  return {
    key: AgentToolsType.EnterPlanMode,
    label: (
      <ToolHeader
        toolName={AgentToolsType.EnterPlanMode}
        stats={t('agent.settings.tooling.permissionMode.plan.title', 'Plan Mode')}
        variant="collapse-label"
        showStatus={false}
      />
    ),
    children: output ? <div>{output}</div> : null
  }
}
