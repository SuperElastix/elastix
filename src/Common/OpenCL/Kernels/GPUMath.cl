/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
int round_half_integer_to_even_32(float x)
{
  if(x >= 0.0)
  {
    x += 0.5f;
  }
  else
  {
    x -= 0.5f;
  }

  const int r = (int)(x);
  return ( x != (float)(r) ) ? r : (int)( 2 * (r / 2) );
}

int round_half_integer_up_32(float x)
{
  return round_half_integer_to_even_32(2 * x + 0.5f) >> 1;
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
int round_half_integer_up(const float v)
{
  float x = v + 0.5;
  int   r = (int)x;

  return (x >= 0.0) ?
         r :
         ( x == (float)(r) ? r : r - (int)(1) );
}
