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
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
// OpenCL implementation of itk::Math

//------------------------------------------------------------------------------
//template< typename TReturn, typename TInput >
//inline TReturn RoundHalfIntegerToEven_base(TInput x)
//{
//  if ( NumericTraits< TInput >::IsNonnegative(x) )
//  {
//    x += static_cast< TInput >( 0.5 );
//  }
//  else
//  {
//    x -= static_cast< TInput >( 0.5 );
//  }
//
//  const TReturn r = static_cast< TReturn >( x );
//  return ( x != static_cast< TInput >( r ) ) ? r : static_cast< TReturn >( 2 *
// ( r / 2 ) );
//}
//------------------------------------------------------------------------------
// OpenCL implementation of
// itkMathDetail::RoundHalfIntegerToEven_32()
int round_half_integer_to_even_32( float x )
{
  if ( x >= 0.0 )
  {
    x += 0.5f;
  }
  else
  {
    x -= 0.5f;
  }

  const int r = (int)( x );
  return ( x != (float)( r ) ) ? r : (int)( 2 * ( r / 2 ) );
}

int round_half_integer_up_32( float x )
{
  return round_half_integer_to_even_32( 2 * x + 0.5f ) >> 1;
}

//------------------------------------------------------------------------------
//template< typename TReturn, typename TInput >
//inline TReturn RoundHalfIntegerUp_base(TInput x)
//{
//  x += static_cast< TInput >( 0.5 );
//  const TReturn r = static_cast< TReturn >( x );
//  return ( NumericTraits< TInput >::IsNonnegative(x) ) ?
//r :
//  ( x == static_cast< TInput >( r ) ? r : r - static_cast< TReturn >( 1 ) );
//}

//------------------------------------------------------------------------------
// OpenCL implementation of
// itkMathDetail::RoundHalfIntegerUp_base()
//------------------------------------------------------------------------------
int round_half_integer_up( const float v )
{
  float x = v + 0.5;
  int   r = (int)x;

  return ( x >= 0.0 ) ?
         r :
         ( x == (float)( r ) ? r : r - (int)( 1 ) );
}
