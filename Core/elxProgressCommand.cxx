/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "elxProgressCommand.h"
#include "elxlog.h"
#include "itkMath.h" // itk::Math::Round

namespace elastix
{
/**
 * ******************* Constructor ***********************
 */

ProgressCommand::ProgressCommand()
{
  m_StartString = "Progress: ";
  m_EndString = "%";
  m_Tag = 0;
  m_TagIsSet = false;
  m_ObservedProcessObject = nullptr;
  m_NumberOfVoxels = 0;
  m_NumberOfUpdates = 0;

} // end Constructor()


/**
 * ******************* Destructor ***********************
 */

ProgressCommand::~ProgressCommand()
{
  this->DisconnectObserver(m_ObservedProcessObject);

} // end Destructor()


/**
 * ******************* SetUpdateFrequency ***********************
 */

void
ProgressCommand::SetUpdateFrequency(const unsigned long numberOfVoxels, const unsigned long numberOfUpdates)
{
  /** Set the member variables. */
  m_NumberOfVoxels = numberOfVoxels;
  m_NumberOfUpdates = numberOfUpdates;

  /** Make sure we have at least one pixel. */
  if (m_NumberOfVoxels < 1)
  {
    m_NumberOfVoxels = 1;
  }

  /** We cannot update more times than there are pixels. */
  if (m_NumberOfUpdates > m_NumberOfVoxels)
  {
    m_NumberOfUpdates = m_NumberOfVoxels;
  }

  /** Make sure we update at least once. */
  if (m_NumberOfUpdates < 1)
  {
    m_NumberOfUpdates = 1;
  }

} // end SetUpdateFrequency()


/**
 * ******************* ConnectObserver ***********************
 */

void
ProgressCommand::ConnectObserver(itk::ProcessObject * filter)
{
  /** Disconnect from old observed filters. */
  this->DisconnectObserver(m_ObservedProcessObject);

  /** Connect to the new filter. */
  m_Tag = filter->AddObserver(itk::ProgressEvent(), this);
  m_TagIsSet = true;
  m_ObservedProcessObject = filter;

} // end ConnectObserver()


/**
 * ******************* DisconnectObserver ***********************
 */

void
ProgressCommand::DisconnectObserver(itk::ProcessObject * filter)
{
  if (m_TagIsSet)
  {
    filter->RemoveObserver(m_Tag);
    m_TagIsSet = false;
    m_ObservedProcessObject = nullptr;
  }

} // end DisconnectObserver()


/**
 * ******************* Execute ***********************
 */

void
ProgressCommand::Execute(itk::Object * caller, const itk::EventObject & event)
{
  itk::ProcessObject * po = dynamic_cast<itk::ProcessObject *>(caller);
  if (!po)
  {
    return;
  }

  if (typeid(event) == typeid(itk::ProgressEvent))
  {
    this->PrintProgress(po->GetProgress());
  }

} // end Execute()


/**
 * ******************* Execute ***********************
 */

void
ProgressCommand::Execute(const itk::Object * caller, const itk::EventObject & event)
{
  const itk::ProcessObject * po = dynamic_cast<const itk::ProcessObject *>(caller);
  if (!po)
  {
    return;
  }

  if (typeid(event) == typeid(itk::ProgressEvent))
  {
    this->PrintProgress(po->GetProgress());
  }

} // end Execute()


/**
 * ******************* PrintProgress ***********************
 */

void
ProgressCommand::PrintProgress(const float progress) const
{
  /** Print the progress to the screen. */
  const int progressInt = itk::Math::Round<float>(100 * progress);

  // Pass the entire message at once, rather than having multiple `<<` insertions.
  const std::string message = '\r' + m_StartString + std::to_string(progressInt) + m_EndString;
  std::cout << message << std::flush;

  /** If the process is completed, print an end-of-line. *
  if ( progress > 0.99999 )
  {
    std::cout << std::endl;
  }*/

} // end PrintProgress()


/**
 * ******************* PrintProgress ***********************
 */

void
ProgressCommand::UpdateAndPrintProgress(const unsigned long currentVoxelNumber) const
{
  const unsigned long frac = static_cast<unsigned long>(m_NumberOfVoxels / m_NumberOfUpdates);
  if (currentVoxelNumber % frac == 0)
  {
    this->PrintProgress(static_cast<float>(currentVoxelNumber) / static_cast<float>(m_NumberOfVoxels));
  }

} // end PrintProgress()


ProgressCommand::Pointer
ProgressCommand::CreateAndSetUpdateFrequency(const unsigned long numberOfVoxels)
{
  const auto result = Self::New();
  result->Self::SetUpdateFrequency(numberOfVoxels, numberOfVoxels);
  result->SetStartString("  Progress: ");
  return result;
}

ProgressCommand::Pointer
ProgressCommand::CreateAndConnect(itk::ProcessObject & processObject)
{
  const auto result = Self::New();
  result->Self::ConnectObserver(&processObject);
  result->Self::SetStartString("  Progress: ");
  result->Self::SetEndString("%");
  return result;
}


} // namespace elastix
