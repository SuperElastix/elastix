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
#include "itkOpenCLImageFormat.h"

namespace itk
{
OpenCLImageFormat::OpenCLImageFormat()
{
  this->m_ImageType                      = 0;
  this->m_Format.image_channel_order     = 0;
  this->m_Format.image_channel_data_type = 0;
}


//------------------------------------------------------------------------------
OpenCLImageFormat::OpenCLImageFormat(
  const OpenCLImageFormat::ChannelOrder channelOrder,
  const OpenCLImageFormat::ChannelType channelType )
{
  this->m_Format.image_channel_order     = channelOrder;
  this->m_Format.image_channel_data_type = channelType;
}


//------------------------------------------------------------------------------
OpenCLImageFormat::OpenCLImageFormat(
  const OpenCLImageFormat::ImageType imageType,
  const OpenCLImageFormat::ChannelOrder channelOrder,
  const OpenCLImageFormat::ChannelType channelType )
{
  this->m_ImageType                      = imageType;
  this->m_Format.image_channel_order     = channelOrder;
  this->m_Format.image_channel_data_type = channelType;
}


//------------------------------------------------------------------------------
bool
OpenCLImageFormat::IsNull() const
{
  return ( this->m_ImageType == 0
         && this->m_Format.image_channel_order == 0
         && this->m_Format.image_channel_data_type == 0 );
}


//------------------------------------------------------------------------------
OpenCLImageFormat::ImageType
OpenCLImageFormat::GetImageType() const
{
  return OpenCLImageFormat::ImageType( this->m_ImageType );
}


//------------------------------------------------------------------------------
OpenCLImageFormat::ChannelOrder
OpenCLImageFormat::GetChannelOrder() const
{
  return OpenCLImageFormat::ChannelOrder( this->m_Format.image_channel_order );
}


//------------------------------------------------------------------------------
OpenCLImageFormat::ChannelType
OpenCLImageFormat::GetChannelType() const
{
  return OpenCLImageFormat::ChannelType( this->m_Format.image_channel_data_type );
}


//------------------------------------------------------------------------------
//! Operator ==
bool
operator==( const OpenCLImageFormat & lhs, const OpenCLImageFormat & rhs )
{
  if( &rhs == &lhs )
  {
    return true;
  }

  return ( lhs.GetImageType() == rhs.GetImageType()
         && lhs.GetChannelOrder() == rhs.GetChannelOrder()
         && lhs.GetChannelType() == rhs.GetChannelType() );
}


//------------------------------------------------------------------------------
//! Operator !=
bool
operator!=( const OpenCLImageFormat & lhs, const OpenCLImageFormat & rhs )
{
  return !( lhs == rhs );
}


} // end namespace itk
